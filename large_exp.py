import cupy as cp
import time

def allocate_memory_on_device(device_id, memory_size):
    cp.cuda.Device(device_id).use()
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)
    num_bytes = memory_size
    d_memory = cp.empty(num_bytes, dtype=cp.uint8)
    d_memory.fill(1)
    device = cp.cuda.Device()
    return d_memory

def run_for_days(days):
    
    device_id0 = 6
    memory_size0 = 30 * 1024 * 1024 * 1024 
    allocated_memory0 = allocate_memory_on_device(device_id0, memory_size0)
    '''
    device_id1 = 0
    memory_size1 = 24 * 1024 * 1024 * 1024 
    allocated_memory1 = allocate_memory_on_device(device_id1, memory_size1)
    
    device_id2 = 1
    memory_size2 = 27 * 1024 * 1024 * 1024 
    allocated_memory2 = allocate_memory_on_device(device_id2, memory_size2)
    '''
    try:

        seconds_to_run = days * 24 * 60 * 60
        start_time = time.time()
        current_time = start_time


        while (current_time - start_time) < seconds_to_run:
            time.sleep(3600) 
            current_time = time.time()
        #print("Time's up. Exiting program...")

    finally:

        del allocated_memory0
        #del allocated_memory1
        #del allocated_memory2
        #cp.cuda.Device(device_id0).synchronize()
        #cp.cuda.Device(device_id1).synchronize()
        #cp.cuda.Device(device_id2).synchronize()


#device_id = 0
#memory_size = 1024 * 1024  # 1MB
days_to_run = 7


run_for_days(days_to_run)