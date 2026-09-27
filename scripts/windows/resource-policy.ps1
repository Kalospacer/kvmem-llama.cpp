$ErrorActionPreference = 'Stop'
if (-not ('KVMemResourcePolicy.Native' -as [type])) {
    Add-Type -TypeDefinition @"
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
namespace KVMemResourcePolicy {
    public sealed class ThreadState { public uint Id; public long Created; public uint Memory; }
    public sealed class ProcessState {
        public int Id; public long Created; public uint Cpu; public uint Memory;
        public List<ThreadState> Threads = new List<ThreadState>();
    }
    public static class Native {
        [DllImport("kernel32.dll", SetLastError=true)] static extern IntPtr OpenProcess(uint access, bool inherit, int id);
        [DllImport("kernel32.dll", SetLastError=true)] static extern IntPtr OpenThread(uint access, bool inherit, uint id);
        [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr h);
        [DllImport("kernel32.dll", SetLastError=true)] static extern uint GetPriorityClass(IntPtr h);
        [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetPriorityClass(IntPtr h, uint value);
        [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetProcessInformation(IntPtr h, int kind, out uint value, uint size);
        [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetProcessInformation(IntPtr h, int kind, ref uint value, uint size);
        [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetThreadInformation(IntPtr h, int kind, out uint value, uint size);
        [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetThreadInformation(IntPtr h, int kind, ref uint value, uint size);
        [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetProcessTimes(IntPtr h, out long c, out long e, out long k, out long u);
        [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetThreadTimes(IntPtr h, out long c, out long e, out long k, out long u);
        [DllImport("kernel32.dll", SetLastError=true)] static extern uint GetProcessIdOfThread(IntPtr h);
        static void Check(bool ok, string api) { if (!ok) throw new Win32Exception(Marshal.GetLastWin32Error(), api); }
        static long Creation(IntPtr h, bool thread) {
            long c,e,k,u;
            Check(thread ? GetThreadTimes(h,out c,out e,out k,out u) : GetProcessTimes(h,out c,out e,out k,out u), "Get times");
            return c;
        }
        static IntPtr ProcessHandle(int id, bool write) {
            IntPtr h=OpenProcess(write ? 0x0600u : 0x0400u,false,id);
            Check(h!=IntPtr.Zero,"OpenProcess"); return h;
        }
        public static ProcessState Capture(int id) {
            IntPtr h=ProcessHandle(id,false);
            try {
                ProcessState s=new ProcessState(); s.Id=id; s.Created=Creation(h,false);
                s.Cpu=GetPriorityClass(h); Check(s.Cpu!=0,"GetPriorityClass");
                Check(GetProcessInformation(h,0,out s.Memory,4),"GetProcessInformation");
                using (Process p=Process.GetProcessById(id)) foreach(ProcessThread t in p.Threads) {
                    IntPtr th=OpenThread(0x0040,false,(uint)t.Id);
                    if (th==IntPtr.Zero) {
                        int error=Marshal.GetLastWin32Error();
                        if(error==87) continue;
                        throw new Win32Exception(error,"OpenThread query");
                    }
                    try {
                        if(GetProcessIdOfThread(th)!=(uint)id) continue;
                        ThreadState ts=new ThreadState();ts.Id=(uint)t.Id;ts.Created=Creation(th,true);
                        Check(GetThreadInformation(th,0,out ts.Memory,4),"GetThreadInformation");s.Threads.Add(ts);
                    } finally { CloseHandle(th); }
                }
                return s;
            } finally {CloseHandle(h);}
        }
        public static void Set(ProcessState before, bool restore) {
            IntPtr h=ProcessHandle(before.Id,true);
            try {
                if(Creation(h,false)!=before.Created) throw new InvalidOperationException("Process identity changed");
                uint cpu=restore ? before.Cpu : 0x20u;
                uint mem=restore ? before.Memory : 5u;
                Check(SetPriorityClass(h,cpu),"SetPriorityClass");
                Check(SetProcessInformation(h,0,ref mem,4),"SetProcessInformation");
                foreach(ThreadState ts in before.Threads) {
                    IntPtr th=OpenThread(0x0060,false,ts.Id);
                    if(th==IntPtr.Zero) {
                        int error=Marshal.GetLastWin32Error();if(error==87) continue;
                        throw new Win32Exception(error,"OpenThread set");
                    }
                    try {
                        if(GetProcessIdOfThread(th)!=(uint)before.Id || Creation(th,true)!=ts.Created) continue;
                        uint value=restore ? ts.Memory : 5u;
                        Check(SetThreadInformation(th,0,ref value,4),"SetThreadInformation");
                        uint actual;Check(GetThreadInformation(th,0,out actual,4),"GetThreadInformation verify");
                        if(actual!=value) throw new InvalidOperationException("Thread policy verification failed");
                    } finally {CloseHandle(th);}
                }
                uint verified;Check(GetProcessInformation(h,0,out verified,4),"GetProcessInformation verify");
                if(GetPriorityClass(h)!=cpu || verified!=mem) throw new InvalidOperationException("Process policy verification failed");
            } finally {CloseHandle(h);}
        }
    }
}
"@
}

function Set-KVMemCurrentResourcePolicy {
    $state = [KVMemResourcePolicy.Native]::Capture($PID)
    try {
        [KVMemResourcePolicy.Native]::Set($state, $false)
    } catch {
        [KVMemResourcePolicy.Native]::Set($state, $true)
        throw
    }
    $env:KVMEM_WINDOWS_RESOURCE_POLICY = 'normal'
}
