import numpy as np
from scipy.optimize import linprog, milp
import random
import time

class MolecularSolverOptimized:
    def __init__(self, random_seed=None):
        # 如果提供了种子，使用它；否则使用当前时间
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        else:
            # 使用时间戳作为种子以确保每次运行都不同
            current_seed = int(time.time() * 1000) % 10000
            random.seed(current_seed)
            np.random.seed(current_seed)
            print(f"使用随机种子: {current_seed}")
        
        # 定义分子及其原子组成 [C, H, O, N]
        self.molecules = {
            'CH4': [1, 4, 0, 0],
            'CO': [1, 0, 1, 0],
            'CO2': [1, 0, 2, 0],
            'H2': [0, 2, 0, 0],
            'H2O': [0, 2, 1, 0],
            'N2': [0, 0, 0, 2],
            'NH3': [0, 3, 0, 1]
        }
        
        # 创建原子-分子矩阵
        self.atom_matrix = np.array([
            [1, 1, 1, 0, 0, 0, 0],  # C
            [4, 0, 0, 2, 2, 0, 3],  # H
            [0, 1, 2, 0, 1, 0, 0],  # O
            [0, 0, 0, 0, 0, 2, 1]   # N
        ])
        
        self.molecule_names = list(self.molecules.keys())
    
    def exact_solution(self, c, h, o, n):
        """使用整数线性规划找到精确解"""
        target_atoms = np.array([c, h, o, n])
        A_eq = self.atom_matrix
        b_eq = target_atoms
        
        # 定义变量界限 (非负整数)
        bounds = [(0, None) for _ in range(len(self.molecules))]
        
        # 随机目标函数，以获得不同的可行解
        c_obj = np.random.rand(len(self.molecules))
        
        # 解整数线性规划问题
        try:
            integrality = np.ones(len(self.molecules))  # 全部变量都是整数
            result = milp(c=c_obj, constraints=[
                {"A": A_eq, "b": b_eq, "type": "=="}
            ], integrality=integrality, bounds=bounds)
            
            if result.success:
                return result.x.astype(int), 0
            else:
                return self.approximate_solution(c, h, o, n)
        except:
            # 如果精确解失败，尝试近似解
            return self.approximate_solution(c, h, o, n)
    
    def approximate_solution(self, c, h, o, n):
        """使用线性规划找到近似解，然后进行四舍五入和调整"""
        target_atoms = np.array([c, h, o, n])
        num_atoms = 4
        num_molecules = len(self.molecules)
        
        # 构建优化问题
        # 我们最小化误差变量 (e_plus 和 e_minus)
        c_obj = np.zeros(num_molecules + 2 * num_atoms)
        c_obj[:num_molecules] = np.random.rand(num_molecules) * 0.01  # 添加小的随机权重
        c_obj[num_molecules:] = 1 + np.random.rand(2 * num_atoms) * 0.01  # 误差变量的权重
        
        # 约束条件：A_eq * x = b_eq
        A_eq = np.zeros((num_atoms, num_molecules + 2 * num_atoms))
        A_eq[:, :num_molecules] = self.atom_matrix
        for i in range(num_atoms):
            A_eq[i, num_molecules + i] = 1           # e_plus[i]
            A_eq[i, num_molecules + num_atoms + i] = -1  # e_minus[i]
        
        b_eq = target_atoms
        
        # 变量界限
        bounds = [(0, None) for _ in range(num_molecules + 2 * num_atoms)]
        
        # 解线性规划问题
        result = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            # 提取分子数量并随机四舍五入
            solution = np.zeros(num_molecules, dtype=int)
            for i in range(num_molecules):
                val = result.x[i]
                # 随机四舍五入 - 概率性地向上或向下取整
                if random.random() < (val - int(val)):
                    solution[i] = int(val) + 1
                else:
                    solution[i] = int(val)
            
            # 计算误差
            error = np.sum(np.abs(np.dot(self.atom_matrix, solution) - target_atoms))
            return solution, error
        else:
            # 如果线性规划失败，使用贪婪算法
            return self.greedy_solution(c, h, o, n)
    
    def greedy_solution(self, c, h, o, n):
        """使用贪婪算法快速找到一个可行解"""
        target_atoms = np.array([c, h, o, n])
        solution = np.zeros(len(self.molecules), dtype=int)
        remaining_atoms = target_atoms.copy()
        
        # 随机分子优先级
        priority_order = list(range(len(self.molecules)))
        random.shuffle(priority_order)
        
        # 按优先级顺序添加分子
        for mol_idx in priority_order:
            mol_atoms = self.atom_matrix[:, mol_idx]
            
            # 计算可以添加的最大分子数量
            if np.any(mol_atoms > 0):
                max_possible = min(
                    remaining_atoms[i] // mol_atoms[i] 
                    for i in range(4) if mol_atoms[i] > 0
                )
            else:
                max_possible = 0
            
            # 随机选择要添加的数量
            if max_possible > 0:
                amount = random.randint(0, max_possible)
                solution[mol_idx] = amount
                remaining_atoms -= mol_atoms * amount
                
        error = np.sum(remaining_atoms)
        return solution, error
    
    def random_solution(self, c, h, o, n, num_tries=5000):
        """生成多个随机解并选择最好的"""
        target_atoms = np.array([c, h, o, n])
        best_solution = None
        best_error = float('inf')
        
        for _ in range(num_tries):
            # 生成随机解
            solution = np.zeros(len(self.molecules), dtype=int)
            remaining = target_atoms.copy()
            
            # 随机顺序添加分子
            mol_indices = list(range(len(self.molecules)))
            random.shuffle(mol_indices)
            
            for mol_idx in mol_indices:
                mol_atoms = self.atom_matrix[:, mol_idx]
                
                # 计算可以添加的最大分子数量
                if np.any(mol_atoms > 0):
                    max_possible = min(
                        remaining[i] // mol_atoms[i] 
                        for i in range(4) if mol_atoms[i] > 0
                    )
                else:
                    max_possible = 0
                
                # 随机选择添加数量
                if max_possible > 0:
                    amount = random.randint(0, max_possible)
                    solution[mol_idx] = amount
                    remaining -= mol_atoms * amount
            
            # 计算误差
            error = np.sum(remaining)
            if error < best_error:
                best_error = error
                best_solution = solution
                
                # 如果找到完美解，直接返回
                if error == 0:
                    break
        
        return best_solution, best_error
    
    def solve(self, c, h, o, n, method='random'):
        """使用指定方法找到解"""
        if method == 'exact':
            print(f"running exact_solution")
            return self.exact_solution(c, h, o, n)
        elif method == 'approximate':
            print(f"running approximate_solution")
            return self.approximate_solution(c, h, o, n)
        elif method == 'greedy':
            print(f"running greedy_solution")
            return self.greedy_solution(c, h, o, n)
        elif method == 'random':
            print(f"running random_solution")
            return self.random_solution(c, h, o, n)
        else:
            # 默认尝试所有方法
            methods = [
                ('exact', self.exact_solution),
                ('random', self.random_solution),
                ('greedy', self.greedy_solution),
                ('approximate', self.approximate_solution)
            ]
            # 随机排序方法以增加多样性
            random.shuffle(methods)
            
            best_solution = None
            best_error = float('inf')
            
            for name, method_func in methods:
                solution, error = method_func(c, h, o, n)
                if error < best_error:
                    best_error = error
                    best_solution = solution
                    
                    # 如果找到完美解，直接返回
                    if error == 0:
                        break
            
            return best_solution, best_error
    
    def format_solution(self, solution):
        """格式化解的输出"""
        result = {}
        for i, molecule in enumerate(self.molecule_names):
            if solution[i] > 0:
                result[molecule] = solution[i]
        return result
    
    def calculate_atoms(self, solution):
        """计算给定解所使用的原子数量"""
        return np.dot(self.atom_matrix, solution)


def solve_molecular_distribution(c, h, o, n, method='exact', random_seed=None):
    """对外暴露的主函数"""
    solver = MolecularSolverOptimized(random_seed)
    solution, error = solver.solve(c, h, o, n, method)
    
    result = {
        "solution": solver.format_solution(solution),
        "used_atoms": {
            "C": solver.calculate_atoms(solution)[0],
            "H": solver.calculate_atoms(solution)[1],
            "O": solver.calculate_atoms(solution)[2],
            "N": solver.calculate_atoms(solution)[3]
        },
        "target_atoms": {"C": c, "H": h, "O": o, "N": n},
        "error": error
    }
    
    return result

# # 使用示例 (不采用main函数)
# c = 24
# h = 24
# o = 48
# n = 48

# # 不指定种子，使用当前时间作为种子
# result = solve_molecular_distribution(c, h, o, n, method='exact')

# print("\n最佳分子组合:")
# for molecule, count in result["solution"].items():
#     print(f"{molecule}: {count}个")

# print(f"\n使用的原子: C={result['used_atoms']['C']}, H={result['used_atoms']['H']}, O={result['used_atoms']['O']}, N={result['used_atoms']['N']}")
# print(f"目标原子: C={result['target_atoms']['C']}, H={result['target_atoms']['H']}, O={result['target_atoms']['O']}, N={result['target_atoms']['N']}")
# print(f"剩余原子: C={result['target_atoms']['C']-result['used_atoms']['C']}, H={result['target_atoms']['H']-result['used_atoms']['H']}, O={result['target_atoms']['O']-result['used_atoms']['O']}, N={result['target_atoms']['N']-result['used_atoms']['N']}")
# print(f"适应度(误差): {result['error']}")

# # 如果想要多次运行得到不同解，可以这样：
# print("\n\n另一个随机解:")
# result2 = solve_molecular_distribution(c, h, o, n)
# for molecule, count in result2["solution"].items():
#     print(f"{molecule}: {count}个")
# print(f"\n使用的原子: C={result['used_atoms']['C']}, H={result['used_atoms']['H']}, O={result['used_atoms']['O']}, N={result['used_atoms']['N']}")
# print(f"目标原子: C={result['target_atoms']['C']}, H={result['target_atoms']['H']}, O={result['target_atoms']['O']}, N={result['target_atoms']['N']}")
# print(f"剩余原子: C={result['target_atoms']['C']-result['used_atoms']['C']}, H={result['target_atoms']['H']-result['used_atoms']['H']}, O={result['target_atoms']['O']-result['used_atoms']['O']}, N={result['target_atoms']['N']-result['used_atoms']['N']}")
# print(f"适应度(误差): {result['error']}")
