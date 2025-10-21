import os
import platform

def get_system_max_memory():
    """
    Returns the total system memory in MB.
    Supports Linux, Windows, and macOS.
    """
    try:
        import psutil
        mem_bytes = psutil.virtual_memory().total
        return int(mem_bytes / (1024 * 1024))
    except ImportError:
        system = platform.system()
        # Linux: Try os.sysconf or /proc/meminfo
        if system == "Linux":
            if hasattr(os, 'sysconf'):
                if 'SC_PAGE_SIZE' in os.sysconf_names and 'SC_PHYS_PAGES' in os.sysconf_names:
                    pages = os.sysconf('SC_PHYS_PAGES')
                    page_size = os.sysconf('SC_PAGE_SIZE')
                    mem_bytes = pages * page_size
                    return int(mem_bytes / (1024 * 1024))
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            mem_kb = int(line.split()[1])
                            return int(mem_kb / 1024)
            except Exception:
                pass
        # Windows: Use ctypes to get memory
        elif system == "Windows":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                mem_bytes = stat.ullTotalPhys
                return int(mem_bytes / (1024 * 1024))
            except Exception:
                pass
        # macOS: Use sysctl
        elif system == "Darwin":
            try:
                import subprocess
                mem_bytes = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']).strip())
                return int(mem_bytes / (1024 * 1024))
            except Exception:
                pass
        # Default to 4000 MB if detection fails
        return 4000

def get_system_cpu_info():
    """
    Returns a dictionary with CPU info:
    {
        'physical_cores': int,
        'logical_cores': int,
        'threads_per_core': int
    }
    Supports Linux, Windows, and macOS.
    """
    info = {
        'physical_cores': 1,
        'logical_cores': 1,
        'threads_per_core': 1
    }
    try:
        import psutil
        info['physical_cores'] = psutil.cpu_count(logical=False)
        info['logical_cores'] = psutil.cpu_count(logical=True)
        if info['physical_cores'] and info['logical_cores']:
            info['threads_per_core'] = info['logical_cores'] // info['physical_cores']
        return info
    except ImportError:
        system = platform.system()
        # Linux: parse /proc/cpuinfo
        if system == "Linux":
            try:
                physical = set()
                logical = 0
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('processor'):
                            logical += 1
                        elif line.startswith('core id'):
                            physical.add(line.strip())
                info['logical_cores'] = logical
                info['physical_cores'] = os.cpu_count() or 1
                if info['physical_cores'] and info['logical_cores']:
                    info['threads_per_core'] = info['logical_cores'] // info['physical_cores']
            except Exception:
                info['physical_cores'] = os.cpu_count() or 1
                info['logical_cores'] = info['physical_cores']
                info['threads_per_core'] = 1
        # Windows: use os.cpu_count
        elif system == "Windows":
            info['physical_cores'] = os.cpu_count() or 1
            info['logical_cores'] = info['physical_cores']
            info['threads_per_core'] = 1
        # macOS: use os.cpu_count
        elif system == "Darwin":
            info['physical_cores'] = os.cpu_count() or 1
            info['logical_cores'] = info['physical_cores']
            info['threads_per_core'] = 1
        return info

def setup_molecule(atomic_numbers, positions, charge=0, spin=1, max_memory=None):
    """
    Shared molecule setup for PySCF-based calculators.
    Converts atomic numbers and positions to PySCF Mole object.
    max_memory is set to system max memory if not provided.
    """
    try:
        from pyscf import gto
        from periodictable import elements
    except ImportError:
        raise ImportError("PySCF and periodictable are required for molecule setup.")

    atomic_symbols = [elements[int(z)].symbol for z in atomic_numbers]
    atom_str = []
    for i, symbol in enumerate(atomic_symbols):
        x, y, z = positions[i]
        atom_str.append(f"{symbol} {x:.6f} {y:.6f} {z:.6f}")

    mol = gto.Mole()
    mol.atom = '\n'.join(atom_str)
    mol.charge = charge
    mol.spin = spin - 1
    mol.unit = 'Angstrom'
    mol.verbose = 0
    if max_memory is None:
        mol.max_memory = get_system_max_memory()
    else:
        mol.max_memory = max_memory
    mol.build()
    return mol
