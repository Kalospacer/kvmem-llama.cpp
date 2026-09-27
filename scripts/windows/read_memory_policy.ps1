# Read-only policy inspection. No process priority, working-set, or task changes.
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class MemoryPolicyReadOnly {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool CloseHandle(IntPtr handle);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool GetProcessWorkingSetSizeEx(IntPtr handle, out UIntPtr min, out UIntPtr max, out uint flags);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool GetProcessInformation(IntPtr handle, int informationClass, out uint data, uint size);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool IsProcessInJob(IntPtr handle, IntPtr job, out bool inJob);
}
"@

$target = Get-Process llama-kvmem-server
$chain = @()
$currentId = $target.Id
for ($depth = 0; $depth -lt 3 -and $currentId -gt 0; $depth++) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$currentId"
    if (-not $proc) { break }
    $runtime = Get-Process -Id $currentId
    $handle = [MemoryPolicyReadOnly]::OpenProcess(0x0400, $false, $currentId)
    if ($handle -eq [IntPtr]::Zero) { throw "OpenProcess query failed: $currentId" }
    try {
        [UIntPtr]$minimum = [UIntPtr]::Zero
        [UIntPtr]$maximum = [UIntPtr]::Zero
        [uint32]$flags = 0
        [uint32]$priority = 0
        [bool]$inJob = $false
        $wsOk = [MemoryPolicyReadOnly]::GetProcessWorkingSetSizeEx($handle, [ref]$minimum, [ref]$maximum, [ref]$flags)
        $priorityOk = [MemoryPolicyReadOnly]::GetProcessInformation($handle, 0, [ref]$priority, 4)
        $jobOk = [MemoryPolicyReadOnly]::IsProcessInJob($handle, [IntPtr]::Zero, [ref]$inJob)
        $chain += [pscustomobject]@{
            pid=$currentId; name=$proc.Name; parent_pid=$proc.ParentProcessId;
            started=$runtime.StartTime.ToString('o'); cpu_priority=$runtime.PriorityClass.ToString();
            memory_priority_query_ok=$priorityOk; memory_priority=$priority;
            working_set_query_ok=$wsOk; minimum=$minimum.ToUInt64(); maximum=$maximum.ToUInt64(); flags=$flags;
            job_query_ok=$jobOk; in_job=$inJob
        }
    } finally {
        [void][MemoryPolicyReadOnly]::CloseHandle($handle)
    }
    $currentId = $proc.ParentProcessId
}

$scheduler = New-Object -ComObject Schedule.Service
$scheduler.Connect()
$task = $scheduler.GetFolder('\').GetTask('KVMem-Server')
$settings = $task.Definition.Settings
[pscustomobject]@{
    captured=(Get-Date).ToString('o'); process_chain=$chain;
    task_priority=$settings.Priority; execution_time_limit=$settings.ExecutionTimeLimit;
    restart_count=$settings.RestartCount; restart_interval=$settings.RestartInterval
} | ConvertTo-Json -Depth 5 -Compress
