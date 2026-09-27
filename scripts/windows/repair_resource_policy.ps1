param(
    [ValidateSet('Inspect','Apply','Rollback')][string]$Mode = 'Inspect',
    [Parameter(Mandatory=$true)][string]$ExePath,
    [string]$JournalPath = 'C:\KVMem\deploy\resource-policy-journal.json'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'resource-policy.ps1')

function Save-Journal($Data) {
    $Data | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath ($JournalPath + '.tmp') -Encoding UTF8
    Move-Item -LiteralPath ($JournalPath + '.tmp') -Destination $JournalPath -Force
}
function Restore-State($Saved) {
    $state = New-Object KVMemResourcePolicy.ProcessState
    $state.Id = $Saved.Id; $state.Created = $Saved.Created
    $state.Cpu = $Saved.Cpu; $state.Memory = $Saved.Memory
    foreach ($t in $Saved.Threads) {
        $thread = New-Object KVMemResourcePolicy.ThreadState
        $thread.Id=$t.Id; $thread.Created=$t.Created; $thread.Memory=$t.Memory
        $state.Threads.Add($thread)
    }
    [KVMemResourcePolicy.Native]::Set($state, $true)
}
if ($Mode -eq 'Rollback') {
    $journal = Get-Content -LiteralPath $JournalPath -Raw | ConvertFrom-Json
    foreach ($state in $journal.before) { Restore-State $state }
    $journal.status = 'rolled_back'; Save-Journal $journal
    return
}

$expected = [IO.Path]::GetFullPath($ExePath)
$targets = @(Get-Process llama-kvmem-server -ErrorAction Stop | Where-Object { $_.Path -eq $expected })
if ($targets.Count -ne 1) { throw 'Expected exactly one process matching the production executable path' }
$target = $targets[0]
$parentInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$($target.Id)"
$parent = Get-Process -Id $parentInfo.ParentProcessId -ErrorAction Stop
if ($parent.ProcessName -notin @('powershell','pwsh') -or $parent.StartTime -gt $target.StartTime) {
    throw 'Unexpected service launcher identity'
}
$states = @([KVMemResourcePolicy.Native]::Capture($target.Id), [KVMemResourcePolicy.Native]::Capture($parent.Id))
if ($Mode -eq 'Inspect') { $states | ConvertTo-Json -Depth 6 -Compress; return }
if (Test-Path -LiteralPath $JournalPath) { throw 'Journal already exists; choose a fresh journal path' }
$journal = [ordered]@{captured=(Get-Date).ToString('o');exe=$expected;status='prepared';before=$states;after=@()}
Save-Journal $journal
try {
    foreach ($state in $states) { [KVMemResourcePolicy.Native]::Set($state, $false) }
    for ($scan=0; $scan -lt 2; $scan++) {
        foreach ($state in $states) {
            $after = [KVMemResourcePolicy.Native]::Capture($state.Id)
            if ($after.Created -ne $state.Created) { throw 'Process identity changed during policy repair' }
            if ($after.Cpu -ne 32 -or $after.Memory -ne 5 -or @($after.Threads | Where-Object Memory -ne 5).Count) {
                throw 'Policy readback failed or a worker explicitly lowered its memory priority'
            }
        }
    }
    $journal.after=@($states | ForEach-Object { [KVMemResourcePolicy.Native]::Capture($_.Id) })
    $journal.status='applied'; Save-Journal $journal
    $journal.after | ForEach-Object { [pscustomobject]@{pid=$_.Id;cpu=$_.Cpu;memory=$_.Memory;threads=$_.Threads.Count;all_threads_normal=(@($_.Threads | Where-Object Memory -ne 5).Count -eq 0)} } | ConvertTo-Json -Compress
} catch {
    $failure = $_
    $rollbackErrors = @()
    foreach ($state in $states) {
        try { [KVMemResourcePolicy.Native]::Set($state, $true) } catch { $rollbackErrors += $_.Exception.Message }
    }
    $journal.status = if ($rollbackErrors.Count) { 'rollback_failed' } else { 'rolled_back' }
    $journal['rollback_errors'] = $rollbackErrors
    Save-Journal $journal
    throw $failure
}
