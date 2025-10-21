import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# FATORAÇÃO DE KHATRI-RAO POR MÍNIMOS QUADRADOS (LSKRF)
# PROBLEMA 02: ANÁLISE DE MONTE CARLO COM NÚMEROS COMPLEXOS
# ===========================================================================

class LSKRF_MonteCarlo:
    """
    Análise robusta de LSKRF com Monte Carlo em ambiente ruidoso.
    
    Parâmetros:
    -----------
    L : int
        Número de realizações de Monte Carlo (padrão: 1000)
    R : int
        Rank da fatoração (padrão: 6)
    max_iter : int
        Máximo de iterações no algoritmo ALS (padrão: 50)
    """

    def __init__(self, L=1000, R=6, max_iter=50):
        self.L = L
        self.R = R
        self.max_iter = max_iter
        
        # Definição de SNR em dB e conversão para linear
        self.snr_db = np.array([0, 5, 10, 15, 20, 25, 30])
        self.SNR_lin = 10**(self.snr_db / 10)
        
        print("="*70)
        print("FATORAÇÃO DE KHATRI-RAO POR MÍNIMOS QUADRADOS (LSKRF)")
        print("PROBLEMA 02 - ANÁLISE DE MONTE CARLO COM RUÍDO")
        print("="*70)
        print(f"\nParâmetros:")
        print(f"  • Número de experimentos (L): {self.L}")
        print(f"  • Rank (R): {self.R}")
        print(f"  • Máx. iterações: {self.max_iter}")
        print(f"  • SNR [dB]: {self.snr_db}")
        print(f"  • Elementos: números complexos (C)")

    # -------------------------------------------------------
    # Função: Produto de Khatri-Rao
    # -------------------------------------------------------
    def _khatri_rao(self, A, B):
        """
        Computa o produto de Khatri-Rao (column-wise Kronecker product).
        
        X = A ⋄ B, onde X(:, r) = kron(A(:, r), B(:, r))
        
        Parâmetros:
        -----------
        A : ndarray, shape (I, R)
            Primeira matriz
        B : ndarray, shape (J, R)
            Segunda matriz
            
        Retorno:
        --------
        X : ndarray, shape (I*J, R)
            Produto de Khatri-Rao
        """
        I, R = A.shape
        J, _ = B.shape
        X = np.zeros((I * J, R), dtype=complex)
        
        for r in range(R):
            # Kronecker de coluna r: kron(A[:, r], B[:, r])
            X[:, r] = np.kron(A[:, r], B[:, r])
        
        return X

    # -------------------------------------------------------
    # Função: Algoritmo LSKRF (Alternating Least Squares)
    # -------------------------------------------------------
    def _lskrf_fit(self, X, I, J):
        """
        Estima A e B resolvendo: min_{A,B} ||X - A ⋄ B||_F^2
        
        Usa Alternating Least Squares (ALS):
        - Fixa A e resolve para B
        - Fixa B e resolve para A
        - Itera até convergência
        
        Parâmetros:
        -----------
        X : ndarray, shape (I*J, R)
            Matriz de dados (possivelmente ruidosa)
        I : int
            Dimensão da matriz A
        J : int
            Dimensão da matriz B
            
        Retorno:
        --------
        A_est, B_est : ndarrays
            Matrizes estimadas
        X_est : ndarray
            Reconstrução X = A_est ⋄ B_est
        """
        X_vec = X.reshape(-1, 1)
        R = X.shape[1]
        
        # Inicialização aleatória
        A_est = np.random.randn(I, R) + 1j * np.random.randn(I, R)
        B_est = np.random.randn(J, R) + 1j * np.random.randn(J, R)

        for iteration in range(self.max_iter):
            
            # ============================================
            # Passo 1: Atualizar B_est (fixando A_est)
            # ============================================
            for r in range(R):
                # Construir matriz de design M_A para coluna r
                # X[:, r] = [A[0,r]*B[:,r]; A[1,r]*B[:,r]; ...; A[I-1,r]*B[:,r]]
                M_A = np.zeros((I * J, J), dtype=complex)
                for i in range(I):
                    M_A[i*J:(i+1)*J, :] = A_est[i, r] * np.eye(J)
                
                # Resolver: B[:, r] = argmin ||X[:, r] - M_A * B[:, r]||^2
                B_est[:, r] = lstsq(M_A, X[:, r])[0]

            # ============================================
            # Passo 2: Atualizar A_est (fixando B_est)
            # ============================================
            for r in range(R):
                # Construir matriz de design M_B para coluna r
                # X[:, r] = [A[0,r]*B[0,r]; A[0,r]*B[1,r]; ...; A[I-1,r]*B[J-1,r]]
                M_B = np.zeros((I * J, I), dtype=complex)
                for j in range(J):
                    M_B[j::J, :] = B_est[j, r] * np.eye(I)
                
                # Resolver: A[:, r] = argmin ||X[:, r] - M_B * A[:, r]||^2
                A_est[:, r] = lstsq(M_B, X[:, r])[0]

        # Reconstruir X
        X_est = self._khatri_rao(A_est, B_est)
        
        return A_est, B_est, X_est

    # -------------------------------------------------------
    # Função: NMSE (Normalized Mean Squared Error)
    # -------------------------------------------------------
    def _nmse(self, X_true, X_est):
        """
        Calcula erro quadrático médio normalizado.
        
        NMSE = ||X_true - X_est||_F^2 / ||X_true||_F^2
        """
        numerator = np.linalg.norm(X_true - X_est, 'fro') ** 2
        denominator = np.linalg.norm(X_true, 'fro') ** 2
        return numerator / denominator

    # -------------------------------------------------------
    # Função: Gerar ruído com SNR especificado
    # -------------------------------------------------------
    def _generate_noisy_data(self, X0, snr_linear):
        """
        Gera versão ruidosa X = X0 + αV com SNR especificado.
        
        SNR_dB = 10 * log10(||X0||_F^2 / (α * ||V||_F^2))
        
        Parâmetros:
        -----------
        X0 : ndarray
            Sinal original (sem ruído)
        snr_linear : float
            SNR em escala linear (não dB)
            
        Retorno:
        --------
        X : ndarray
            Sinal ruidoso
        V : ndarray
            Ruído aditivo (para referência)
        """
        # Gerar ruído Gaussiano complexo
        V = np.random.randn(*X0.shape) + 1j * np.random.randn(*X0.shape)
        
        # Calcular fator α para atingir SNR desejado
        # SNR = ||X0||_F^2 / (α^2 * ||V||_F^2)
        # α = ||X0||_F / (sqrt(SNR) * ||V||_F)
        
        norm_X0_sq = np.linalg.norm(X0, 'fro') ** 2
        norm_V_sq = np.linalg.norm(V, 'fro') ** 2
        alpha = np.sqrt(norm_X0_sq / (snr_linear * norm_V_sq))
        
        # Aplicar fator ao ruído
        V_scaled = alpha * V
        
        # Sinal ruidoso
        X = X0 + V_scaled
        
        return X, V_scaled

    # -------------------------------------------------------
    # Função: Análise de Monte Carlo
    # -------------------------------------------------------
    def _run_monte_carlo(self, config):
        """
        Executa L experimentos de Monte Carlo para uma configuração.
        
        Retorna NMSE médio para X, A, B em cada nível de SNR.
        """
        I = config['I']
        J = config['J']
        
        # Arrays para armazenar NMSE de todos os experimentos
        nmse_X_all = np.zeros((len(self.snr_db), self.L))
        nmse_A_all = np.zeros((len(self.snr_db), self.L))
        nmse_B_all = np.zeros((len(self.snr_db), self.L))

        for ll in range(self.L):
            for ii, snr_val in enumerate(self.SNR_lin):
                # ========================
                # Gerar fatores originais
                # ========================
                A_true = np.random.randn(I, self.R) + 1j * np.random.randn(I, self.R)
                B_true = np.random.randn(J, self.R) + 1j * np.random.randn(J, self.R)
                
                # Produto de Khatri-Rao (sem ruído)
                X0 = self._khatri_rao(A_true, B_true)

                # ========================
                # Adicionar ruído
                # ========================
                X_noisy, _ = self._generate_noisy_data(X0, snr_val)

                # ========================
                # Estimar A e B
                # ========================
                A_est, B_est, X_est = self._lskrf_fit(X_noisy, I, J)

                # ========================
                # Calcular NMSE
                # ========================
                nmse_X_all[ii, ll] = self._nmse(X0, X_est)
                nmse_A_all[ii, ll] = self._nmse(A_true, A_est)
                nmse_B_all[ii, ll] = self._nmse(B_true, B_est)

            # Feedback de progresso
            if (ll + 1) % 200 == 0:
                print(f"    ✓ Completado {ll + 1}/{self.L} experimentos")

        # Calcular média sobre os L experimentos
        nmse_X_mean = np.mean(nmse_X_all, axis=1)
        nmse_A_mean = np.mean(nmse_A_all, axis=1)
        nmse_B_mean = np.mean(nmse_B_all, axis=1)

        return nmse_X_mean, nmse_A_mean, nmse_B_mean

    # -------------------------------------------------------
    # Função: Plotar Resultados
    # -------------------------------------------------------
    def _plot_results(self, configs, results):
        """
        Plota curvas NMSE vs SNR para ambas as configurações.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        colors = {'X': '#1f77b4', 'A': '#ff7f0e', 'B': '#2ca02c'}
        markers = {'X': 'o', 'A': 's', 'B': '^'}

        for idx, (config, result) in enumerate(zip(configs, results)):
            ax = axes[idx]
            nmse_X, nmse_A, nmse_B = result

            # Plot NMSE vs SNR em escala log
            ax.semilogy(self.snr_db, nmse_X, marker=markers['X'], linewidth=2.5, 
                       markersize=8, label='X (reconstrução)', color=colors['X'])
            ax.semilogy(self.snr_db, nmse_A, marker=markers['A'], linewidth=2.5, 
                       markersize=8, label='A (fator)', color=colors['A'])
            ax.semilogy(self.snr_db, nmse_B, marker=markers['B'], linewidth=2.5, 
                       markersize=8, label='B (fator)', color=colors['B'])

            # Configurar eixos e labels
            ax.set_xlabel('SNR [dB]', fontsize=13, fontweight='bold')
            ax.set_ylabel('NMSE', fontsize=13, fontweight='bold')
            ax.set_title(f"Configuração {idx+1}: (I,J) = ({config['I']},{config['J']}), R = {self.R}",
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=11, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xticks(self.snr_db)

        fig.suptitle(f'LSKRF - Análise de Monte Carlo ({self.L} experimentos)\nDados Complexos', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig('lskrf_problema02_monte_carlo.png', dpi=150, bbox_inches='tight')
        plt.show()

    # -------------------------------------------------------
    # Função: Análise e Discussão
    # -------------------------------------------------------
    def _print_analysis(self, configs, results):
        """
        Imprime análise dos resultados e discussões.
        """
        print("\n" + "="*70)
        print("RESULTADOS E ANÁLISE")
        print("="*70)

        for idx, (config, result) in enumerate(zip(configs, results)):
            nmse_X, nmse_A, nmse_B = result
            
            print(f"\n{'CONFIGURAÇÃO ' + str(idx+1):^70}")
            print(f"  Dimensões: (I,J) = ({config['I']},{config['J']}), R = {self.R}")
            print(f"  Matriz resultante: X ∈ C^({config['I']*config['J']}×{self.R})")
            print("-" * 70)
            
            # Tabela de resultados
            print(f"{'SNR [dB]':<12} {'NMSE(X)':<18} {'NMSE(A)':<18} {'NMSE(B)':<18}")
            print("-" * 70)
            for ii, snr in enumerate(self.snr_db):
                print(f"{snr:<12.1f} {nmse_X[ii]:<18.4e} {nmse_A[ii]:<18.4e} {nmse_B[ii]:<18.4e}")

            # Análise qualitativa
            print("\nOBSERVAÇÕES:")
            
            # SNR baixo
            print(f"  • SNR baixo ({self.snr_db[0]} dB): NMSE ≈ {nmse_X[0]:.2e}")
            print(f"    → Ruído domina a estimação")
            
            # SNR médio
            idx_mid = len(self.snr_db) // 2
            print(f"  • SNR médio ({self.snr_db[idx_mid]} dB): NMSE ≈ {nmse_X[idx_mid]:.2e}")
            print(f"    → Regime de transição")
            
            # SNR alto
            print(f"  • SNR alto ({self.snr_db[-1]} dB): NMSE ≈ {nmse_X[-1]:.2e}")
            print(f"    → Regime onde sinal domina")
            
            # Melhoria com aumento de SNR
            improvement = nmse_X[0] / nmse_X[-1]
            print(f"  • Melhoria SNR 0→30 dB: {improvement:.2e}x")

    # -------------------------------------------------------
    # Função: Executar Análise Completa
    # -------------------------------------------------------
    def run(self):
        """Executa análise completa de Monte Carlo."""
        
        # Definir configurações
        configs = [
            {'I': 10, 'J': 10},
            {'I': 30, 'J': 10}
        ]
        
        results = []
        
        for idx, config in enumerate(configs, 1):
            print(f"\n{'─'*70}")
            print(f"Processando Configuração {idx}: (I,J) = ({config['I']},{config['J']})")
            print(f"{'─'*70}")
            
            result = self._run_monte_carlo(config)
            results.append(result)

        # Plotar resultados
        print("\n" + "─"*70)
        print("Gerando gráficos...")
        self._plot_results(configs, results)

        # Análise e discussão
        self._print_analysis(configs, results)

        print("\n" + "="*70)
        print("✓ ANÁLISE CONCLUÍDA COM SUCESSO")
        print("="*70)
        print("\nArquivos gerados:")
        print("  • lskrf_problema02_monte_carlo.png")
        print("\n" + "="*70 + "\n")


# ===========================================================================
# MAIN  
# ===========================================================================

if __name__ == "__main__":
    # Criar e executar análise de Monte Carlo
    mc_analysis = LSKRF_MonteCarlo(L=1000, R=6, max_iter=50)
    mc_analysis.run()