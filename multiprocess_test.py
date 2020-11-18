import multiprocessing as mp
print(mp.cpu_count())
POISON_PILL = "death"


def main_processor(main_queue, sub_queue_dict):
    while True:
        queue_input = main_queue.get()
        if queue_input == POISON_PILL:
            return
        else:
            val = queue_input["value"]
            id = queue_input["id"]
            out_queue = sub_queue_dict[id]
            out_queue.put(val**2)


def sub_processor(main_queue, sub_queue, value, lock):
    sub_dict = {"id": value, "value": value}
    main_queue.put(sub_dict)
    ret_val = sub_queue.get()
    lock.acquire()
    print(value, " ^2 = ", ret_val)
    lock.release()
    return


if __name__ == "__main__":
    queue_manager = mp.Manager()
    main_queue = queue_manager.Queue()
    lock = mp.Lock()
    sub_processes = []
    sub_queue_dict = {}
    for idx in range(32):
        sub_queue = queue_manager.Queue()
        sub_process = mp.Process(target=sub_processor, args=(main_queue, sub_queue, idx, lock))
        sub_queue_dict[idx] = sub_queue
        sub_processes.append(sub_process)
    main_process = mp.Process(target=main_processor, args=(main_queue, sub_queue_dict))
    main_process.start()
    for sub_process in sub_processes:
            sub_process.start()
    for sub_process in sub_processes:
        sub_process.join()
    main_queue.put(POISON_PILL)
    main_process.join()



