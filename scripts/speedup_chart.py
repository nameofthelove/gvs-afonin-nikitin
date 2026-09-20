import json
import os
import matplotlib.pyplot as plt

def main():
    json_path = 'benchmark_results.json'
    if not os.path.exists(json_path):
        print(f"Ошибка: файл {json_path} не найден. Сначала запустите бенчмарки.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    cpu_times = {}
    gpu_times = {}

    for bench in data.get('benchmarks', []):
        name = bench['name']
        parts = name.split('/')
        if len(parts) < 2:
            continue

        size = int(parts[1])
        time_ns = bench['real_time']

        if 'BM_VectorAdd_CPU' in parts[0]:
            cpu_times[size] = time_ns
        elif 'BM_VectorAdd_GPU' in parts[0]:
            gpu_times[size] = time_ns

    # Находим общие размеры векторов N
    common_sizes = sorted(list(set(cpu_times.keys()) & set(gpu_times.keys())))
    speedups = [cpu_times[s] / gpu_times[s] for s in common_sizes]

    plt.figure(figsize=(9, 5))
    plt.plot(common_sizes, speedups, 'o-', color='#7570b3', linewidth=2, label='Ускорение S(N)')
    plt.xscale('log')

    # Пороговая линия паритета (1x)
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Паритет (1x)')

    plt.xlabel('Размер вектора N (число элементов)', fontsize=11)
    plt.ylabel('Коэффициент ускорения S(N) = T_CPU / T_GPU', fontsize=11)
    plt.title('График ускорения GPU относительно CPU', fontsize=12, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    os.makedirs('docs/images', exist_ok=True)
    out_path = 'docs/images/speedup_chart.png'
    plt.savefig(out_path, dpi=300)
    print(f"График ускорения сохранен: {out_path}")

if __name__ == '__main__':
    main()