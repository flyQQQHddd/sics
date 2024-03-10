from multiprocessing import Pool
from matplotlib import pyplot as plt
import pandas as pd
from typing import List
import numpy as np
import random
from autonavi import autonavi_distance_from_coord, autonavi_coordination, autonavi_generate_map

MAX_TIME = 200
NUMBER_OF_INDIVIDUAL = 200
pIntercross = 0.95
pMutate = 0.2

DISTANCE_TYPE = 0 # 0:直线距离，1:驾车导航距离


def distance_matrix(points: pd.DataFrame) -> np.ndarray:

    # 创建距离矩阵
    distances = np.zeros((len(points), len(points)))

    # 多进程计算两地之间的距离
    p = Pool()

    results = []
    for idx_begin, begin in points.iterrows():

        for idx_end, end in points.iterrows():

            if idx_begin == idx_end:
                continue

            result = p.apply_async(

                autonavi_distance_from_coord,
                (
                    (begin['lon'], begin['lat']),
                    (end['lon'], end['lat']),
                    DISTANCE_TYPE,
                    (idx_begin, idx_end)
                )
            )

            results.append(result)

    p.close()
    p.join()

    # 将计算结果存储在距离矩阵中
    for result in results:
        idx_begin = result.get()['flag'][0]
        idx_end = result.get()['flag'][1]
        distance = result.get()['distance']

        distances[idx_begin - 1, idx_end - 1] = distance

    return distances


def get_coordination(points: pd.DataFrame) -> pd.DataFrame:

    # 多进程调用 API
    p = Pool()
    
    results = []
    for idx, name in enumerate(points['name']):
        result = p.apply_async(autonavi_coordination, (name, idx))
        results.append(result)

    p.close()
    p.join()

    # 处理多进程返回的结果
    lons = np.zeros(len(points))
    lats = np.zeros(len(points))
    for result in results:

        name = result.get()['name']
        coord: str = result.get()['coordination']
        idx = result.get()['flag']
        lons[idx] = float(coord.split(',')[0])
        lats[idx] = float(coord.split(',')[1])
        print(f'{name}: {coord}')

    points['lat'] = lats
    points['lon'] = lons

    return points


def draw_poi_map(points: pd.DataFrame, filename: str) -> None:

    # 生成点位的图示（高德地图API规范的字符串）
    markers = ''
    for idx, point in points.iterrows():
        markers += f"large,0xFF0000,{idx}:{point['lon']},{point['lat']}|"
    markers = markers.rstrip('|')

    # 生成路径的图示（高德地图API规范的字符串）
    paths = '5,0x0000ff,1,,:'
    origin_flag = False
    origin = None
    for idx, point in points.iterrows():
        paths += f"{point['lon']},{point['lat']};"
        if origin_flag is False:
            origin_flag = True
            origin = f"{point['lon']},{point['lat']}"

    paths += origin
    
    # 生成地图
    response = autonavi_generate_map(markers, paths)

    print(response)

    with open(filename, 'wb') as f:
        f.write(response)


def _fitness(code, dis_mat):
    beg = code
    end = np.roll(code, -1)

    value = 0
    for i, j in list(zip(beg, end)):
        value += dis_mat[i][j]

    value = 1 / value

    return value


def _distance(code, dis_mat):
    beg = code
    end = np.roll(code, -1)

    value = 0
    for i, j in list(zip(beg, end)):
        value += dis_mat[i][j]

    return value


def fitness(population: List[np.ndarray], dis_mat):
    fitness_value = []

    for code in population:

        beg = code
        end = np.roll(code, -1)

        value = 0
        for i, j in list(zip(beg, end)):
            value += dis_mat[i][j]

        value = 1 / value

        fitness_value.append(value)

    return fitness_value


def select(population: List[np.ndarray], fitness_value):

    sel = random.choices(population, fitness_value, k=len(population))

    return sel


def intercross(population: List[np.ndarray], pval):
    population.append(population[0])
    out_population = []

    # 次序杂交法
    for idx in range(len(population) - 1):

        if random.random() > pval:
            out_population.append(population[idx])
            continue

        parent = population[idx]
        parent2 = population[idx + 1]

        beg_pos = random.randint(1, len(population[0]) - 2)
        end_pos = random.randint(beg_pos + 1, len(population[0]) - 1)

        # 从第一个父个体中截取子序列
        child = parent[beg_pos:end_pos + 1]

        # 第二个父个体从end_pos开始排序
        parent2 = np.roll(parent2, -(end_pos + 1))

        # 去掉第二个父个体中的重复项
        del_idx = []
        for idx, val in enumerate(parent2):
            if val in child:
                del_idx.append(idx)

        parent2 = np.delete(parent2, del_idx)

        # 拼接
        child = np.concatenate([child, parent2])
        child = np.roll(child, beg_pos)

        out_population.append(child)

    return out_population


def mutate(population: List[np.ndarray], pval):

    # 互换变异
    for item in population:

        if random.random() > pval:
            continue

        beg_pos = random.randint(1, len(population[0]) - 2)
        end_pos = random.randint(beg_pos + 1, len(population[0]) - 1)

        # 交换
        item[[beg_pos, end_pos]] = item[[end_pos, beg_pos]]

    return population


def reverse(population: List[np.ndarray], dis_mat):
    out_population = []

    # 进化逆转
    for item in population:

        beg_pos = random.randint(1, len(population[0]) - 3)
        end_pos = random.randint(beg_pos + 1, len(population[0]) - 1)

        # 交换
        recerse_list = list(range(beg_pos, end_pos + 1))
        recerse_list.reverse()

        old_item = item.copy()
        old_fitness = _fitness(item, dis_mat)
        item[list(reversed(recerse_list))] = item[recerse_list]
        new_fitness = _fitness(item, dis_mat)

        # 选择保留适应值大的个体
        if old_fitness > new_fitness:
            out_population.append(old_item)
        else:
            out_population.append(item)

    return population


if __name__ == "__main__":

    # 读取文件，查询经纬度
    points = pd.read_csv('./poi.csv', index_col=0)
    points = get_coordination(points)

    # 计算距离矩阵
    dis_mat = distance_matrix(points)

    # 种群数组
    population = []

    # 随机生成初始种群
    for i in range(NUMBER_OF_INDIVIDUAL):
        code = np.random.permutation(np.arange(len(points)))
        population.append(code)


    # 最优个体记录
    best = None
    best_fitness = 0
    best_fitness_record = []

    for t in range(MAX_TIME):

        # 计算适应度
        fitness_value = fitness(population, dis_mat)

        # 记录最优适应度，以及历史最优适应度
        if max(fitness_value) > best_fitness:
            best_fitness = max(fitness_value)
            best_idx = fitness_value.index(best_fitness)
            best = population[best_idx]
        best_fitness_record.append(best_fitness)

        # 输出迭代情况
        mean_fitness = np.mean(fitness_value)
        print(f'time:{t} | best:{best} | best fitness:{best_fitness} | mean fitness:{mean_fitness}')

        # 父体选择
        population = select(population, fitness_value)

        # 交叉
        population = intercross(population, pIntercross)

        # 变异
        population = mutate(population, pMutate)

        # 逆转
        population = reverse(population, dis_mat)


    print(f'求解路径：{best + 1}')
    print(f'求解距离：{_distance(best, dis_mat)}')

    path = np.zeros(len(points), np.int32)
    for i in range(len(points)):
        path[best[i]] = i
    points['path'] = path

    points = points.sort_values(by='path', ascending=True)

    # -------------------------------------------------------------
    # 绘制 POI 地图
    draw_poi_map(points, './output/path.png')

    # -------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(best_fitness_record)
    fig.savefig('./output/train.png', dpi=500)
