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

    cpu_sizes, cpu_times = [], []
    gpu_sizes, gpu_times = [], []

    for bench in data.get('benchmarks', []):
        name = bench['name']
        parts = name.split('/')
        if len(parts) < 2:
            continue

        size = int(parts[1])
        time_ms = bench['real_time'] / 1e6  # Перевод из наносекунд в миллисекунды

        if 'BM_VectorAdd_CPU' in parts[0]:
            cpu_sizes.append(size)
            cpu_times.append(time_ms)
        elif 'BM_VectorAdd_GPU' in parts[0]:
            gpu_sizes.append(size)
            gpu_times.append(time_ms)

    plt.figure(figsize=(9, 5))
    plt.loglog(cpu_sizes, cpu_times, 'o-', label='CPU (std::vector)', color='#d95f02', linewidth=2)
    plt.loglog(gpu_sizes, gpu_times, 's-', label='GPU (CUDA)', color='#1b9e77', linewidth=2)

    plt.xlabel('Размер вектора N (число элементов)', fontsize=11)
    plt.ylabel('Время выполнения T(N), мс', fontsize=11)
    plt.title('Зависимость времени выполнения от размера вектора T(N)', fontsize=12, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    os.makedirs('docs/images', exist_ok=True)
    out_path = 'docs/images/complexity_chart.png'
    plt.savefig(out_path, dpi=300)
    print(f"График временной сложности сохранен: {out_path}")

if __name__ == '__main__':
    main()