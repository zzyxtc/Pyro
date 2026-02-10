#include <linux/module.h>
#include <linux/kprobes.h>
#include <linux/kallsyms.h>
#include <linux/thermal.h>
#include <linux/topology.h>
#include <linux/kernel.h>
#include <asm/msr.h>
#include <linux/netlink.h>
#include <linux/skbuff.h>
#include <net/sock.h>
#include <linux/workqueue.h>
#include <linux/kmod.h>
#include <linux/cpu.h>
#include <linux/cpufreq.h>
#include <linux/cpumask.h>
#include <linux/perf_event.h>
#include <linux/spinlock.h>
#include <linux/pid.h>

#define NETLINK_USER 31
#define INIT_MSG_TYPE 0x10
#define AVG_TEMP_MSG_TYPE 0x11
#define TEMP_DATA_MSG_TYPE 0x12
#define MAX_PHYSICAL_CORES 16

#define MAX_TEMP_PER_CORE 30
#define STEP_SIZE 10
#define SAMPLE_INTERVAL 100

static void handle_sliding_window(int cpu);
static void free_per_cpu_buffers(int cpu_num);
static int read_pcie_power(u64 *power);
static void process_perf_data(struct work_struct *work);
static void send_temperatures_work(struct work_struct *work);
static void nl_recv_msg(struct sk_buff *skb);

/* Enum for perf event types */
enum my_perf_event_type {
    EVENT_CPU_CYCLES,
    EVENT_INSTRUCTIONS,
    EVENT_CACHE_REFS,
    EVENT_CACHE_MISSES,
    EVENT_BUS_CYCLES,
    EVENT_BRANCH_INSTRUCTIONS,
    EVENT_BRANCH_MISSES,
    EVENT_TASK_CLOCK,
    EVENT_L1_DATA_READ,
    EVENT_L1_DATA_WRITE,
    EVENT_L1_DATA_MISS,
    EVENT_L1_INS_MISS,
    EVENT_L2_READ_MISS,
    EVENT_CONTEXT_SWITCHES,
    EVENT_PAGE_FAULTS,
    EVENT_TYPE_MAX
};

/* Global variables */
static DEFINE_SPINLOCK(python_start_lock);
static struct perf_event **perf_events[EVENT_TYPE_MAX];
static struct workqueue_struct *my_wq;
static struct work_struct my_work;
static struct workqueue_struct *perf_wq;
static struct work_struct perf_work;

static struct kprobe kp;
static struct sock *nl_sk = NULL;

struct temperature_data {
    int logical_core_id;
    int physical_core_id;
    u8 coretemp;
    u32 pkg_energy;
    u32 dram_energy;
    u8 cpu_freq;
    u8 cpu_voltage;
    u8 cpu_load;
    u64 instruction_diff;
    u64 cycle_diff;
    u64 cache_diff;
    u64 cache_miss_diff;
    u64 bus_cycle_diff;
    u64 branch_instruction_diff;
    u32 branch_insMiss_diff;
    u32 task_clock_diff;
    u32 l1_data_read_diff;
    u32 l1_data_write_diff;
    u32 l1_data_miss_diff;
    u32 l1_ins_miss_diff;
    u32 l2_read_miss_diff;
    u32 context_switches_diff;
    u32 page_faults_diff;
}__attribute__((packed));

struct core_data {
    int temp_sum;
    u32 prev_pkg_energy;
    u32 prev_dram_energy;
    u64 prev_aperf;
    u64 prev_mperf;
    int sample_count;
    bool need_process;
    u64 prev_values[EVENT_TYPE_MAX];
} __cacheline_aligned;

static struct temperature_data **temp_storage_per_cpu;
static int *temp_count_per_cpu;
static int cpu_num;
static bool python_started = false;
static pid_t python_pid = 0;
static bool isTrain = false;
static struct core_data *core_datas;

// 添加模块参数
static pid_t user_pid = 0;
module_param(user_pid, int, 0);
MODULE_PARM_DESC(user_pid, "PID of the Python process");

/* Function to create perf event based on type */
static struct perf_event* create_perf_event(enum my_perf_event_type type, int cpu) {
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(struct perf_event_attr));
    
    attr.size = sizeof(struct perf_event_attr);
    attr.disabled = 0;
    attr.exclude_kernel = 0;
    
    switch (type) {
        case EVENT_CPU_CYCLES:
            attr.type = PERF_TYPE_HARDWARE;
            attr.config = PERF_COUNT_HW_CPU_CYCLES;
            break;
        case EVENT_INSTRUCTIONS:
            attr.type = PERF_TYPE_HARDWARE;
            attr.config = PERF_COUNT_HW_INSTRUCTIONS;
            break;
        case EVENT_CACHE_REFS:
            attr.type = PERF_TYPE_HARDWARE;
            attr.config = PERF_COUNT_HW_CACHE_REFERENCES;
            break;
        case EVENT_CACHE_MISSES:
            attr.type = PERF_TYPE_HARDWARE;
            attr.config = PERF_COUNT_HW_CACHE_MISSES;
            break;
        case EVENT_BUS_CYCLES:
            attr.type = PERF_TYPE_HARDWARE;
            attr.config = PERF_COUNT_HW_BUS_CYCLES;
            break;
        case EVENT_BRANCH_INSTRUCTIONS:
            attr.type = PERF_TYPE_HARDWARE;
            attr.config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
            break;
        case EVENT_BRANCH_MISSES:
            attr.type = PERF_TYPE_HARDWARE;
            attr.config = PERF_COUNT_HW_BRANCH_MISSES;
            break;
        case EVENT_TASK_CLOCK:
            attr.type = PERF_TYPE_SOFTWARE;
            attr.config = PERF_COUNT_SW_TASK_CLOCK;
            break;
        case EVENT_L1_DATA_READ:
            attr.type = PERF_TYPE_HW_CACHE;
            attr.config = PERF_COUNT_HW_CACHE_L1D |(PERF_COUNT_HW_CACHE_OP_READ << 8) |(PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
            break;
        case EVENT_L1_DATA_WRITE:
            attr.type = PERF_TYPE_HW_CACHE;
            attr.config = PERF_COUNT_HW_CACHE_L1D |(PERF_COUNT_HW_CACHE_OP_WRITE << 8) |(PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
            break;
        case EVENT_L1_DATA_MISS:
            attr.type = PERF_TYPE_HW_CACHE;
            attr.config = PERF_COUNT_HW_CACHE_L1D |(PERF_COUNT_HW_CACHE_OP_READ << 8) |(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
            break;
        case EVENT_L1_INS_MISS:
            attr.type = PERF_TYPE_HW_CACHE;
            attr.config = PERF_COUNT_HW_CACHE_L1I |(PERF_COUNT_HW_CACHE_OP_READ << 8) |(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
            break;
        case EVENT_L2_READ_MISS:
            attr.type = PERF_TYPE_RAW;
            attr.config = 0x24 | (0x3f << 8);
            break;
        case EVENT_CONTEXT_SWITCHES:
            attr.type = PERF_TYPE_SOFTWARE;
            attr.config = PERF_COUNT_SW_CONTEXT_SWITCHES;
            break;
        case EVENT_PAGE_FAULTS:
            attr.type = PERF_TYPE_SOFTWARE;
            attr.config = PERF_COUNT_SW_PAGE_FAULTS;
            break;
        default:
            return NULL;
    }

    return perf_event_create_kernel_counter(&attr, cpu, NULL, NULL, NULL);
}

/* Function to read perf event value */
static u64 read_perf_event(struct perf_event *event) {
    u64 count = 0;
    u64 enabled, running;
    
    if (event) {
        count = perf_event_read_value(event, &enabled, &running);
    }
    return count;
}

/* Function to cleanup perf events */
static void cleanup_perf_events(void) {
    int i, cpu;
    
    for (i = 0; i < EVENT_TYPE_MAX; i++) {
        if (perf_events[i]) {
            for (cpu = 0; cpu < cpu_num; cpu++) {
                if (perf_events[i][cpu]) {
                    perf_event_disable(perf_events[i][cpu]);
                    perf_event_release_kernel(perf_events[i][cpu]);
                    perf_events[i][cpu] = NULL;
                }
            }
            kfree(perf_events[i]);
            perf_events[i] = NULL;
        }
    }
}

static void process_perf_data(struct work_struct *work) {
    int logical_core_id = smp_processor_id();
    struct core_data *cd = &core_datas[logical_core_id];
    u64 current_values[EVENT_TYPE_MAX];
    u64 diff_values[EVENT_TYPE_MAX];
    int i;
    u32 low, high;
    u8 cpu_voltage, freq, load;
    
    if (!cd->need_process) return;
    
    u64 msr_val;
    rdmsrl_on_cpu(logical_core_id, MSR_IA32_PERF_STATUS, &msr_val);
    freq = (msr_val >> 8) & 0xFF;
    cpu_voltage = (msr_val >> 32) & 0xFFFF;
    
    u32 current_energy;
    if(rdmsr_safe_on_cpu(logical_core_id, MSR_PKG_ENERGY_STATUS, &current_energy, &high)){
        printk(KERN_ERR "CPU %d: Failed to read MSR_PKG_ENERGY_STATUS\n", logical_core_id); return ;
    }
    u32 prev_energy = cd->prev_pkg_energy;
    u64 delta_energy;
    if (current_energy >= prev_energy) {
        delta_energy = current_energy - prev_energy;
    } else {
        delta_energy = (0xFFFFFFFF - prev_energy) + current_energy + 1;
    }
    
    u32 current_dram_energy;
    if(rdmsr_safe_on_cpu(logical_core_id, MSR_DRAM_ENERGY_STATUS, &current_dram_energy, &high)){
        printk(KERN_ERR "CPU %d: Failed to read MSR_DRAM_ENERGY_STATUS\n", logical_core_id); return ;
    }
    
    u64 aperf, mperf;
    u64 delta_aperf, delta_mperf;
    rdmsrl_on_cpu(logical_core_id, MSR_IA32_APERF, &aperf);
    rdmsrl_on_cpu(logical_core_id, MSR_IA32_MPERF, &mperf);
    delta_aperf = aperf - cd->prev_aperf;
    delta_mperf = mperf - cd->prev_mperf + 1;
    if (delta_mperf == 0) {printk(KERN_ERR "Failed to read MSR_IA32_APERF and MSR_IA32_MPERF\n"); return ;}
    load = (delta_aperf * 100) / delta_mperf;
    
    for (i = 0; i < EVENT_TYPE_MAX; i++) {
        current_values[i] = read_perf_event(perf_events[i][logical_core_id]);
        diff_values[i] = current_values[i] - cd->prev_values[i];
        cd->prev_values[i] = current_values[i];
    }

    if (cd->sample_count == 0) {
        printk(KERN_WARNING "sample_count is zero!\n");
        return;
    }
    
    int avg_temp = cd->temp_sum / cd->sample_count;
    
    struct temperature_data *buffer = temp_storage_per_cpu[logical_core_id];
    if (temp_count_per_cpu[logical_core_id] < MAX_TEMP_PER_CORE) {
        buffer[temp_count_per_cpu[logical_core_id]].logical_core_id = logical_core_id;
        buffer[temp_count_per_cpu[logical_core_id]].physical_core_id = topology_core_id(logical_core_id);
        buffer[temp_count_per_cpu[logical_core_id]].coretemp = avg_temp;
        buffer[temp_count_per_cpu[logical_core_id]].pkg_energy = delta_energy;
        buffer[temp_count_per_cpu[logical_core_id]].dram_energy = current_dram_energy - cd->prev_dram_energy;
        buffer[temp_count_per_cpu[logical_core_id]].cpu_freq = freq;
        buffer[temp_count_per_cpu[logical_core_id]].cpu_voltage = cpu_voltage;
        buffer[temp_count_per_cpu[logical_core_id]].cpu_load = load; 
        
        buffer[temp_count_per_cpu[logical_core_id]].cycle_diff = diff_values[EVENT_CPU_CYCLES];
        buffer[temp_count_per_cpu[logical_core_id]].instruction_diff = diff_values[EVENT_INSTRUCTIONS];
        buffer[temp_count_per_cpu[logical_core_id]].cache_diff = diff_values[EVENT_CACHE_REFS];
        buffer[temp_count_per_cpu[logical_core_id]].cache_miss_diff = diff_values[EVENT_CACHE_MISSES];
        buffer[temp_count_per_cpu[logical_core_id]].bus_cycle_diff = diff_values[EVENT_BUS_CYCLES];
        buffer[temp_count_per_cpu[logical_core_id]].branch_instruction_diff = diff_values[EVENT_BRANCH_INSTRUCTIONS];
        buffer[temp_count_per_cpu[logical_core_id]].branch_insMiss_diff = diff_values[EVENT_BRANCH_MISSES];
        buffer[temp_count_per_cpu[logical_core_id]].task_clock_diff = diff_values[EVENT_TASK_CLOCK];
        
        buffer[temp_count_per_cpu[logical_core_id]].l1_data_read_diff = diff_values[EVENT_L1_DATA_READ];
        buffer[temp_count_per_cpu[logical_core_id]].l1_data_write_diff = diff_values[EVENT_L1_DATA_WRITE];
        buffer[temp_count_per_cpu[logical_core_id]].l1_data_miss_diff = diff_values[EVENT_L1_DATA_MISS];
        buffer[temp_count_per_cpu[logical_core_id]].l1_ins_miss_diff = diff_values[EVENT_L1_INS_MISS];
        buffer[temp_count_per_cpu[logical_core_id]].l2_read_miss_diff = diff_values[EVENT_L2_READ_MISS];
        
        buffer[temp_count_per_cpu[logical_core_id]].context_switches_diff = diff_values[EVENT_CONTEXT_SWITCHES];
        buffer[temp_count_per_cpu[logical_core_id]].page_faults_diff = diff_values[EVENT_PAGE_FAULTS];
     
        temp_count_per_cpu[logical_core_id]++;
    }else {
        if (python_pid != 0) {
            queue_work(my_wq, &my_work);
        } else {
            printk(KERN_WARNING "Python PID not set, cannot send temperatures\n");
        }
        handle_sliding_window(logical_core_id);
    }
    
    cd->temp_sum = 0;
    cd->prev_pkg_energy = current_energy;
    cd->prev_dram_energy = current_dram_energy;
    cd->prev_aperf = aperf;
    cd->prev_mperf = mperf;
    cd->sample_count = 0;
    cd->need_process = false;
}

static int allocate_per_cpu_buffers(int cpu_num) {
    int core, i;
    
    temp_storage_per_cpu = kcalloc(cpu_num, sizeof(struct temperature_data *), GFP_KERNEL);
    temp_count_per_cpu = kcalloc(cpu_num, sizeof(int), GFP_KERNEL);
    core_datas = kcalloc(cpu_num, sizeof(struct core_data), GFP_KERNEL);
    
    if (!temp_storage_per_cpu || !temp_count_per_cpu || !core_datas) {
        printk(KERN_ERR "Memory allocation failed for per-CPU buffers\n");
        return -ENOMEM;
    }

    for (core = 0; core < cpu_num; core++) {
        struct temperature_data *buffer = kcalloc(MAX_TEMP_PER_CORE, sizeof(struct temperature_data), GFP_KERNEL);
        if (!buffer) return -ENOMEM;
        temp_storage_per_cpu[core] = buffer;
        temp_count_per_cpu[core] = 0;
        core_datas[core].need_process = false;
        core_datas[core].temp_sum = 0;
        core_datas[core].sample_count = 0;
        rdmsrl_on_cpu(core, MSR_IA32_APERF, &core_datas[core].prev_aperf);
        rdmsrl_on_cpu(core, MSR_IA32_MPERF, &core_datas[core].prev_mperf);
        
        u32 low_energy, high_energy;
        if (rdmsr_safe_on_cpu(core, MSR_PKG_ENERGY_STATUS, &low_energy, &high_energy)) {
            printk(KERN_ERR "CPU %d: Failed to read MSR_PKG_ENERGY_STATUS during init\n", core); return -ENOMEM;
        }
        core_datas[core].prev_pkg_energy = (u64)high_energy << 32 | low_energy;
        
        u32 dram_energy;
        if(rdmsr_safe_on_cpu(core, MSR_DRAM_ENERGY_STATUS, &dram_energy, &high_energy)){
            printk(KERN_ERR "CPU %d: Failed to read MSR_DRAM_ENERGY_STATUS\n", core); return -ENOMEM;
        }
        core_datas[core].prev_dram_energy = dram_energy;
        
        for (i = 0; i < EVENT_TYPE_MAX; i++) {
            core_datas[core].prev_values[i] = 0;
        }
    }
    return 0;
}

static int init_perf_events(void) {
    int i, cpu, ret;
    
    for (i = 0; i < EVENT_TYPE_MAX; i++) {
        perf_events[i] = kcalloc(cpu_num, sizeof(struct perf_event *), GFP_KERNEL);
        if (!perf_events[i]) {
            cleanup_perf_events();
            return -ENOMEM;
        }
    }

    for (cpu = 0; cpu < cpu_num; cpu++) {
        for (i = 0; i < EVENT_TYPE_MAX; i++) {
            perf_events[i][cpu] = create_perf_event((enum my_perf_event_type)i, cpu);
            if (IS_ERR(perf_events[i][cpu])) {
                ret = PTR_ERR(perf_events[i][cpu]);
                perf_events[i][cpu] = NULL;
                cleanup_perf_events();
                return ret;
            }
        }
    }

    return 0;
}

static void handle_sliding_window(int cpu) {
    int retain_count = MAX_TEMP_PER_CORE - STEP_SIZE;
    memmove(temp_storage_per_cpu[cpu], temp_storage_per_cpu[cpu] + STEP_SIZE, retain_count * sizeof(struct temperature_data));
    temp_count_per_cpu[cpu] = retain_count;
}

static void send_temperatures_work(struct work_struct *work) {
    int cpu;
    for (cpu = 0; cpu < cpu_num; cpu++) {
        if (temp_count_per_cpu[cpu] >= MAX_TEMP_PER_CORE && python_pid != 0) {
            int msg_size = sizeof(struct temperature_data) * MAX_TEMP_PER_CORE;
            struct sk_buff *skb = nlmsg_new(NLMSG_SPACE(msg_size), GFP_KERNEL);

            if (!skb) {
                printk(KERN_ERR "Failed to allocate skb for CPU %d.\n", cpu);
                continue;
            }

            struct nlmsghdr *nlh = nlmsg_put(skb, 0, 0, TEMP_DATA_MSG_TYPE, msg_size, 0);
            if (!nlh) {
                kfree_skb(skb);
                printk(KERN_ERR "Failed to create nlmsg for CPU %d.\n", cpu);
                continue;
            }

            memcpy(nlmsg_data(nlh), temp_storage_per_cpu[cpu], msg_size);

            int res = nlmsg_unicast(nl_sk, skb, python_pid);
            if (res < 0)
                printk(KERN_ERR "Error sending message to PID %d: %d\n", python_pid, res);

            handle_sliding_window(cpu);
        }
    }
}


static void handler_post(struct kprobe *p, struct pt_regs *regs, unsigned long flags) {
    int logical_core_id = smp_processor_id();
    struct core_data *cd = &core_datas[logical_core_id];
    u64 MSRval_Tjmax, MSRval_Temp;
    u8 tjunction, delta, coretemp;
    
    if (rdmsrl_safe_on_cpu(logical_core_id, 0x1A2, &MSRval_Tjmax) || 
        rdmsrl_safe_on_cpu(logical_core_id, 0x19C, &MSRval_Temp)) {
        return ;
    }
    
    tjunction = (MSRval_Tjmax >> 16) & 0x7f;
    delta = (MSRval_Temp >> 16) & 0x7f;
    coretemp = tjunction - delta;

    cd->temp_sum += coretemp;
    cd->sample_count++;

    if (cd->sample_count >= SAMPLE_INTERVAL) {
        cd->need_process = true;
        queue_work(perf_wq, &perf_work);
    }
    
    return ;
}

static void nl_recv_msg(struct sk_buff *skb) {
    struct nlmsghdr *nlh = nlmsg_hdr(skb);

    if (nlh->nlmsg_pid == 0) {
        printk(KERN_INFO "Ignoring message from kernel self or invalid PID\n");
        return;
    }

    if (nlh->nlmsg_len < NLMSG_HDRLEN) {
        printk(KERN_ERR "Invalid Netlink message\n");
        return;
    }

    if (nlh->nlmsg_type == INIT_MSG_TYPE) {
        if (nlh->nlmsg_len == NLMSG_HDRLEN) {
            if (user_pid != 0 && nlh->nlmsg_pid != user_pid) {
                printk(KERN_WARNING "Received init message from unexpected PID: %d (expected %d)\n", 
                       nlh->nlmsg_pid, user_pid);
                return;
            }
            python_pid = nlh->nlmsg_pid;
            printk(KERN_INFO "Initialized with Python PID: %d\n", python_pid);
        } else {
            printk(KERN_ERR "Invalid initialization message\n");
        }
    } else if (nlh->nlmsg_type == AVG_TEMP_MSG_TYPE) {
        if (nlh->nlmsg_len >= NLMSG_HDRLEN + sizeof(int) * 3) {
            struct {
                unsigned int logical_core;
                unsigned int physical_core;
                int avg_temp;
            } *data;
            data = (void *)NLMSG_DATA(nlh);
            printk(KERN_INFO "Received average temperature: %d°C from Logical Core: %u, Physical Core: %u\n",
                   data->avg_temp, data->logical_core, data->physical_core);
        } else {
            printk(KERN_ERR "Invalid average temperature message\n");
        }
    } else {
        printk(KERN_INFO "Unknown message type: %d\n", nlh->nlmsg_type);
    }
}

static void free_per_cpu_buffers(int cpu_num) {
    if (temp_storage_per_cpu) {
        int core;
        for (core = 0; core < cpu_num; core++) {
            if (temp_storage_per_cpu[core]) {
                kfree(temp_storage_per_cpu[core]);
                temp_storage_per_cpu[core] = NULL;
            }
        }
        kfree(temp_storage_per_cpu);
        temp_storage_per_cpu = NULL;
    }
    if (temp_count_per_cpu) {
        kfree(temp_count_per_cpu);
        temp_count_per_cpu = NULL;
    }
    if (core_datas) {
        kfree(core_datas);
        core_datas = NULL;
    }
}

static int __init nl_init(void) {
    struct netlink_kernel_cfg cfg = {
        .input = nl_recv_msg,
    };

    nl_sk = netlink_kernel_create(&init_net, NETLINK_USER, &cfg);
    if (!nl_sk) {
        printk(KERN_ERR "Failed to create Netlink socket\n");
        return -ENOMEM;
    }

    cpu_num = num_online_cpus();
    
    int ret = allocate_per_cpu_buffers(cpu_num);
    if (ret < 0) {
        printk(KERN_ERR "Failed to allocate per-CPU buffers\n");
        netlink_kernel_release(nl_sk);
        return ret;
    }
    
    ret = init_perf_events();
    if (ret != 0) {
        printk(KERN_ERR "Failed to create perf events!\n");
        free_per_cpu_buffers(cpu_num);
        netlink_kernel_release(nl_sk);
        return ret;
    }
    
    my_wq = create_workqueue("my_workqueue");
    INIT_WORK(&my_work, send_temperatures_work);
    
    perf_wq = create_workqueue("perf_workqueue");
    INIT_WORK(&perf_work, process_perf_data);

    kp.symbol_name = "hrtimer_interrupt";
    //kp.pre_handler = handler_pre;
    kp.post_handler = handler_post;

    ret = register_kprobe(&kp);
    if (ret < 0) {
        printk(KERN_ERR "Failed to register Kprobe: %d\n", ret);
        cleanup_perf_events();
        free_per_cpu_buffers(cpu_num);
        destroy_workqueue(my_wq);
        destroy_workqueue(perf_wq);
        netlink_kernel_release(nl_sk);
        return ret;
    }
    
    if (user_pid != 0) {
        printk(KERN_INFO "Loaded with Python PID: %d\n", user_pid);
    } else {
        printk(KERN_INFO "Waiting for Python PID via Netlink\n");
    }
    
    printk(KERN_INFO "Netlink module loaded\n");
    return 0;
}

static void __exit nl_exit(void) {
    unregister_kprobe(&kp);
    
    cancel_work_sync(&my_work);
    cancel_work_sync(&perf_work);
    
    flush_workqueue(my_wq);
    flush_workqueue(perf_wq);
    
    destroy_workqueue(my_wq);
    destroy_workqueue(perf_wq);
    
    if (nl_sk) {
        netlink_kernel_release(nl_sk);
        nl_sk = NULL;
    }
    
    cleanup_perf_events();
    free_per_cpu_buffers(cpu_num);

    printk(KERN_INFO "Module unloaded.\n");
}

module_init(nl_init);
module_exit(nl_exit);
MODULE_LICENSE("GPL");
