import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# DECOMPOSIÇÃO EM VALORES SINGULARES VIA PRODUTO DE KRONECKER (KPSVD)
# PROBLEMA 02: APROXIMAÇÃO DE POSTO-r
# ===========================================================================

class KPSVD_RankApproximation:
    """
    Implementação da KPSVD com análise de aproximação de posto-r.
    
    Objetivo:
    ---------
    1. Gerar matriz em blocos X (M=N=P=Q=4)
    2. Calcular KPSVD e posto-Kronecker r_KP
    3. Para vários valores r ≤ r_KP, calcular aproximação de posto-r
    4. Analisar erro de aproximação
    """

    def __init__(self, M=4, N=4, P=4, Q=4):
        """
        Parâmetros:
        -----------
        M, N : int
            Número de blocos (linhas e colunas)
        P, Q : int
            Dimensões de cada bloco X_{i,j} ∈ C^(P×Q)
        """
        self.M = M
        self.N = N
        self.P = P
        self.Q = Q
        
        print("="*80)
        print("DECOMPOSIÇÃO KPSVD - APROXIMAÇÃO DE POSTO-r")
        print("PROBLEMA 02")
        print("="*80)
        print(f"\nConfiguração:")
        print(f"  • M = N = P = Q = {M}")
        print(f"  • Matriz completa X ∈ C^({M*P}×{N*Q}) = C^({M*P}×{N*Q})")
        print(f"  • Total de blocos: {M}×{N} = {M*N}")
        
        # Gerar matriz em blocos
        self._generate_block_matrix()

    # -------------------------------------------------------
    # Função: Gerar Matriz em Blocos
    # -------------------------------------------------------
    def _generate_block_matrix(self):
        """
        Gera matriz em blocos X_{i,j} = randn(P, Q) para todo i,j.
        """
        print("\nGerando matriz em blocos (Gaussiana real)...")
        
        # Criar matriz de blocos
        self.X_blocks = []
        for i in range(self.M):
            row_blocks = []
            for j in range(self.N):
                # Gerar bloco com randn (distribuição normal real)
                block = np.random.randn(self.P, self.Q)
                row_blocks.append(block)
            self.X_blocks.append(row_blocks)
        
        # Converter para matriz completa
        self.X_full = np.block(self.X_blocks)
        
        print(f"✓ Matriz gerada: X ∈ R^({self.X_full.shape[0]}×{self.X_full.shape[1]})")
        print(f"  • Elementos: distribuição Gaussiana real N(0,1)")
        print(f"  • ||X||_F = {np.linalg.norm(self.X_full, 'fro'):.4f}")
        print(f"  • min(X) = {np.min(self.X_full):.4f}")
        print(f"  • max(X) = {np.max(self.X_full):.4f}")

    # -------------------------------------------------------
    # Função: KPSVD - Decomposição
    # -------------------------------------------------------
    def compute_kpsvd(self):
        """
        Computa a decomposição KPSVD completa.
        """
        print("\n" + "-"*80)
        print("COMPUTANDO KPSVD")
        print("-"*80)
        
        # SVD padrão
        print("  1. Aplicando SVD padrão...")
        U_full, S, Vh_full = svd(self.X_full, full_matrices=True)
        
        print(f"     • U ∈ R^({U_full.shape[0]}×{U_full.shape[1]})")
        print(f"     • S: {len(S)} valores singulares")
        print(f"     • V^T ∈ R^({Vh_full.shape[0]}×{Vh_full.shape[1]})")
        
        # Determinar posto de Kronecker
        print("\n  2. Determinando posto de Kronecker...")
        tol = max(self.X_full.shape) * np.finfo(float).eps * S[0]
        r_KP = np.sum(S > tol)
        
        print(f"     • Tolerância: {tol:.4e}")
        print(f"     • σ_max = {S[0]:.4e}")
        print(f"     • σ_min (não-zero) = {S[r_KP-1]:.4e}")
        print(f"     • Posto de Kronecker r_KP = {r_KP}")
        
        # Extrair valores singulares
        print("\n  3. Valores singulares de Kronecker:")
        sigma = S[:r_KP]
        
        # Mostrar todos os valores singulares
        print(f"     • Top 10 valores singulares:")
        for k in range(min(10, r_KP)):
            percentage = (sigma[k]**2 / np.sum(sigma**2)) * 100
            print(f"       σ_{k+1:2d} = {sigma[k]:8.4f}  ({percentage:5.2f}% da energia)")
        
        # Reshape vetores singulares
        print("\n  4. Construindo matrizes U_k e V_k...")
        U_matrices = []
        V_matrices = []
        
        for k in range(r_KP):
            u_k = U_full[:, k]
            v_k = Vh_full[k, :]
            
            U_k = u_k.reshape(self.M, self.P)
            V_k = v_k.reshape(self.N, self.Q)
            
            U_matrices.append(U_k)
            V_matrices.append(V_k)
        
        print(f"     ✓ Construídas {r_KP} matrizes U_k ∈ R^({self.M}×{self.P})")
        print(f"     ✓ Construídas {r_KP} matrizes V_k ∈ R^({self.N}×{self.Q})")
        
        # Armazenar resultados
        self.sigma = sigma
        self.U_matrices = U_matrices
        self.V_matrices = V_matrices
        self.r_KP = r_KP
        
        print("-"*80)
        print("✓ KPSVD computada com sucesso!")
        
        return sigma, U_matrices, V_matrices, r_KP

    # -------------------------------------------------------
    # Função: Aproximação de Posto-r
    # -------------------------------------------------------
    def rank_r_approximation(self, r):
        """
        Calcula a aproximação de posto-r de X.
        
        X_r = Σ_{k=1}^r σ_k (U_k ⊗ V_k)
        
        Parâmetros:
        -----------
        r : int
            Posto da aproximação (r ≤ r_KP)
            
        Retorno:
        --------
        X_r : ndarray
            Aproximação de posto-r
        """
        if r > self.r_KP:
            print(f"⚠ Aviso: r={r} > r_KP={self.r_KP}, usando r={self.r_KP}")
            r = self.r_KP
        
        X_r = np.zeros_like(self.X_full)
        
        for k in range(r):
            term = self.sigma[k] * np.kron(self.U_matrices[k], self.V_matrices[k])
            X_r += term
        
        return X_r

    # -------------------------------------------------------
    # Função: Análise de Aproximação para Vários Postos
    # -------------------------------------------------------
    def analyze_rank_approximations(self):
        """
        Analisa erro de aproximação para vários valores de r.
        """
        print("\n" + "="*80)
        print("ANÁLISE DE APROXIMAÇÕES DE POSTO-r")
        print("="*80)
        
        # Escolher valores de r para analisar
        if self.r_KP <= 10:
            rank_values = list(range(1, self.r_KP + 1))
        else:
            rank_values = [1, 2, 3, 4, 5, 8, 10, self.r_KP//2, self.r_KP]
        
        print(f"\nValores de r analisados: {rank_values}")
        
        # Calcular erros para cada r
        self.rank_results = []
        
        print("\n" + "-"*80)
        print(f"{'r':<5} {'||X - X_r||_F':<18} {'Erro Relativo':<18} {'Energia [%]':<12}")
        print("-"*80)
        
        norm_X = np.linalg.norm(self.X_full, 'fro')
        total_energy = np.sum(self.sigma**2)
        
        for r in rank_values:
            # Calcular aproximação
            X_r = self.rank_r_approximation(r)
            
            # Calcular erro
            error_abs = np.linalg.norm(self.X_full - X_r, 'fro')
            error_rel = error_abs / norm_X
            
            # Calcular energia capturada
            energy_captured = np.sum(self.sigma[:r]**2) / total_energy * 100
            
            # Armazenar resultados
            self.rank_results.append({
                'r': r,
                'X_r': X_r,
                'error_abs': error_abs,
                'error_rel': error_rel,
                'energy': energy_captured
            })
            
            print(f"{r:<5} {error_abs:<18.4e} {error_rel:<18.4e} {energy_captured:<12.2f}")
        
        print("-"*80)
        
        # Análise adicional
        print("\nOBSERVAÇÕES:")
        
        # Encontrar r para 90%, 95%, 99% de energia
        energy_cumsum = np.cumsum(self.sigma**2) / total_energy
        r_90 = np.searchsorted(energy_cumsum, 0.90) + 1
        r_95 = np.searchsorted(energy_cumsum, 0.95) + 1
        r_99 = np.searchsorted(energy_cumsum, 0.99) + 1
        
        print(f"  • Para 90% da energia: r = {r_90} (de {self.r_KP})")
        print(f"  • Para 95% da energia: r = {r_95} (de {self.r_KP})")
        print(f"  • Para 99% da energia: r = {r_99} (de {self.r_KP})")
        
        # Taxa de compressão
        original_params = self.M * self.P * self.N * self.Q
        compressed_params_90 = r_90 * (self.M * self.P + self.N * self.Q + 1)
        compression_90 = original_params / compressed_params_90
        
        print(f"\n  • Taxa de compressão (r={r_90}, 90% energia): {compression_90:.2f}x")
        print(f"    → Parâmetros originais: {original_params}")
        print(f"    → Parâmetros comprimidos: {compressed_params_90}")

    # -------------------------------------------------------
    # Função: Visualização Comparativa
    # -------------------------------------------------------
    def plot_comparisons(self):
        """
        Plota comparações visuais das aproximações.
        """
        fig = plt.figure(figsize=(18, 12))
        
        # Selecionar alguns valores de r para visualizar
        r_values_to_plot = [1, 2, 4, 8, self.r_KP]
        r_values_to_plot = [r for r in r_values_to_plot if r <= self.r_KP]
        
        n_plots = len(r_values_to_plot) + 2  # +2 para original e erro vs r
        
        # ====================================================
        # 1. Matriz Original
        # ====================================================
        ax1 = plt.subplot(3, 3, 1)
        im1 = ax1.imshow(self.X_full, aspect='auto', cmap='RdBu_r', 
                         vmin=-np.max(np.abs(self.X_full)), 
                         vmax=np.max(np.abs(self.X_full)))
        ax1.set_title('X Original', fontsize=12, fontweight='bold')
        ax1.set_xlabel(f'Colunas ({self.N*self.Q})')
        ax1.set_ylabel(f'Linhas ({self.M*self.P})')
        
        # Linhas de blocos
        for i in range(1, self.M):
            ax1.axhline(i * self.P - 0.5, color='black', linewidth=0.5, alpha=0.3)
        for j in range(1, self.N):
            ax1.axvline(j * self.Q - 0.5, color='black', linewidth=0.5, alpha=0.3)
        plt.colorbar(im1, ax=ax1)
        
        # ====================================================
        # 2-6. Aproximações para diferentes r
        # ====================================================
        for idx, r in enumerate(r_values_to_plot[:5]):
            ax = plt.subplot(3, 3, idx + 2)
            
            # Encontrar resultado correspondente
            result = next((res for res in self.rank_results if res['r'] == r), None)
            if result is None:
                X_r = self.rank_r_approximation(r)
                error_rel = np.linalg.norm(self.X_full - X_r, 'fro') / np.linalg.norm(self.X_full, 'fro')
                energy = np.sum(self.sigma[:r]**2) / np.sum(self.sigma**2) * 100
            else:
                X_r = result['X_r']
                error_rel = result['error_rel']
                energy = result['energy']
            
            im = ax.imshow(X_r, aspect='auto', cmap='RdBu_r',
                          vmin=-np.max(np.abs(self.X_full)),
                          vmax=np.max(np.abs(self.X_full)))
            ax.set_title(f'X_{r} (r={r})\nErro: {error_rel:.2e}, Energia: {energy:.1f}%', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Colunas')
            ax.set_ylabel('Linhas')
            plt.colorbar(im, ax=ax)
        
        # ====================================================
        # 7. Erro vs Posto
        # ====================================================
        ax7 = plt.subplot(3, 3, 7)
        errors = [res['error_rel'] for res in self.rank_results]
        ranks = [res['r'] for res in self.rank_results]
        ax7.semilogy(ranks, errors, 'o-', linewidth=2, markersize=8, color='#d62728')
        ax7.set_xlabel('Posto r', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Erro Relativo', fontsize=11, fontweight='bold')
        ax7.set_title('Erro vs Posto da Aproximação', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, linestyle='--')
        
        # ====================================================
        # 8. Valores Singulares
        # ====================================================
        ax8 = plt.subplot(3, 3, 8)
        ax8.semilogy(range(1, len(self.sigma)+1), self.sigma, 'o-', 
                     linewidth=2, markersize=6, color='#1f77b4')
        ax8.set_xlabel('Índice k', fontsize=11, fontweight='bold')
        ax8.set_ylabel('σ_k', fontsize=11, fontweight='bold')
        ax8.set_title('Valores Singulares de Kronecker', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, linestyle='--')
        
        # ====================================================
        # 9. Energia Acumulada
        # ====================================================
        ax9 = plt.subplot(3, 3, 9)
        energy_cumsum = np.cumsum(self.sigma**2) / np.sum(self.sigma**2) * 100
        ax9.plot(range(1, len(energy_cumsum)+1), energy_cumsum, 'o-', 
                linewidth=2, markersize=6, color='#2ca02c')
        ax9.axhline(90, color='r', linestyle='--', alpha=0.5, label='90%')
        ax9.axhline(95, color='orange', linestyle='--', alpha=0.5, label='95%')
        ax9.axhline(99, color='blue', linestyle='--', alpha=0.5, label='99%')
        ax9.set_xlabel('Posto r', fontsize=11, fontweight='bold')
        ax9.set_ylabel('Energia Acumulada [%]', fontsize=11, fontweight='bold')
        ax9.set_title('Energia Capturada vs Posto', fontsize=12, fontweight='bold')
        ax9.legend(fontsize=9)
        ax9.grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle(f'KPSVD - Aproximações de Posto-r (r_KP = {self.r_KP})', 
                    fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('kpsvd_problema02_rank_approximations.png', dpi=150, bbox_inches='tight')
        plt.show()

    # -------------------------------------------------------
    # Função: Conclusões
    # -------------------------------------------------------
    def print_conclusions(self):
        """
        Imprime conclusões sobre aproximações de posto-r.
        """
        print("\n" + "="*80)
        print("CONCLUSÕES - APROXIMAÇÃO DE POSTO-r VIA KPSVD")
        print("="*80)
        
        print(f"""
1. POSTO DE KRONECKER:
   
   • Posto completo: r_KP = {self.r_KP}
   • Matriz: {self.M*self.P}×{self.N*self.Q} = {self.M*self.P}×{self.N*self.Q}
   • Posto de Kronecker captura estrutura em blocos

2. QUALIDADE DA APROXIMAÇÃO:
   
   • Aproximação de posto-r: X_r = Σ_{{k=1}}^r σ_k (U_k ⊗ V_k)
   • Erro relativo decresce com r
   • Teorema de Eckart-Young: X_r é ÓTIMA em norma de Frobenius
   • ||X - X_r||_F é minimizada para dado r

3. COMPRESSÃO:
   
   ✓ Parâmetros originais: {self.M*self.P*self.N*self.Q}
   ✓ Parâmetros posto-r: r × (MP + NQ + 1) 
   ✓ Para r << r_KP: compressão significativa
   ✓ Trade-off entre erro e compressão

4. INTERPRETAÇÃO DOS VALORES SINGULARES:
   
   • σ_k grande → termo importante
   • σ_k pequeno → termo pouco relevante
   • Decaimento rápido → matriz altamente compressível
   • Decaimento lento → necessita muitos termos

5. APLICAÇÕES PRÁTICAS:
   
   ✓ Compressão de dados estruturados
   ✓ Filtragem de ruído (mantendo r termos principais)
   ✓ Redução de dimensionalidade
   ✓ Análise de componentes principais estruturada
   ✓ Regularização em problemas inversos

6. TEOREMA FUNDAMENTAL (Aproximação Ótima):
   
   Para qualquer r < r_KP:
   
   X_r = arg min_{{rank-Kronecker(Y)≤r}} ||X - Y||_F
   
   A aproximação KPSVD de posto-r é a MELHOR aproximação
   possível em termos de norma de Frobenius!
""")
        
        print("="*80 + "\n")

    # -------------------------------------------------------
    # Função: Executar Análise Completa
    # -------------------------------------------------------
    def run(self):
        """Executa análise completa."""
        
        # Computar KPSVD
        self.compute_kpsvd()
        
        # Analisar aproximações
        self.analyze_rank_approximations()
        
        # Visualização
        self.plot_comparisons()
        
        # Conclusões
        self.print_conclusions()


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    # Criar e executar análise
    kpsvd = KPSVD_RankApproximation(M=4, N=4, P=4, Q=4)
    kpsvd.run()