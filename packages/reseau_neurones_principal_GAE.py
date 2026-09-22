import numpy as np
from .adam_update import adam_update
from .relu import relu, relu_derivative
from .parametres_reseau_principal import NB_ACTIONS_POSSIBLE, NB_NEURONES_LAYER1, NB_NEURONES_LAYER2, TAILLE_STATE, coeff_entropie, TAILLE_BATCHS, NB_ENTRAINEMENT_BATCH, learning_rate, MAX_NORME_GRADIENT, EPSILON_CLIP_VALUE, gamma
from .softmax import softmax
from .reseau_neurones_V_estimate import Reseau_neurones_V_estimate
from numpy.typing import NDArray


class Reseau_neurones_principal:
    def __init__(self, nom_fichier: str):
        self.nom_fichier = nom_fichier
        self.NB_ACTIONS_POSSIBLE = NB_ACTIONS_POSSIBLE
        self.NB_NEURONES_LAYER1 = NB_NEURONES_LAYER1
        self.NB_NEURONES_LAYER2 = NB_NEURONES_LAYER2
        self.TAILLE_STATE = TAILLE_STATE

        try:
            data = np.load(nom_fichier)
            self.w1 = data['W1']
            self.w2 = data['W2']
            self.w3 = data['W3']
            self.b1 = data['B1']
            self.b2 = data['B2']
            self.b3 = data['B3']
            self.mW1 = data['mW1']; self.vW1 = data['vW1']
            self.mW2 = data['mW2']; self.vW2 = data['vW2']
            self.mW3 = data['mW3']; self.vW3 = data['vW3']
            self.mB1 = data['mB1']; self.vB1 = data['vB1']
            self.mB2 = data['mB2']; self.vB2 = data['vB2']
            self.mB3 = data['mB3']; self.vB3 = data['vB3']
            self.t_adam = int(data['t_adam'])

        except FileNotFoundError:
            self.w1 = (np.random.randn(self.NB_NEURONES_LAYER1, self.TAILLE_STATE) * np.sqrt(2 / self.TAILLE_STATE)).astype(np.float32)  # He init pour ReLU
            self.w2 = (np.random.randn(self.NB_NEURONES_LAYER2, self.NB_NEURONES_LAYER1) * np.sqrt(2 / self.NB_NEURONES_LAYER1)).astype(np.float32)
            self.w3 = (np.random.randn(self.NB_ACTIONS_POSSIBLE, self.NB_NEURONES_LAYER2) * 0.01).astype(np.float32)            
            self.b1 = np.zeros(self.NB_NEURONES_LAYER1, dtype=np.float32)
            self.b2 = np.zeros(self.NB_NEURONES_LAYER2, dtype=np.float32)
            self.b3 = np.zeros(self.NB_ACTIONS_POSSIBLE, dtype=np.float32)
            self.mW1 = np.zeros_like(self.w1, dtype=np.float32); self.vW1 = np.zeros_like(self.w1, dtype=np.float32)
            self.mW2 = np.zeros_like(self.w2, dtype=np.float32); self.vW2 = np.zeros_like(self.w2, dtype=np.float32)
            self.mW3 = np.zeros_like(self.w3, dtype=np.float32); self.vW3 = np.zeros_like(self.w3, dtype=np.float32)
            self.mB1 = np.zeros_like(self.b1, dtype=np.float32); self.vB1 = np.zeros_like(self.b1, dtype=np.float32)
            self.mB2 = np.zeros_like(self.b2, dtype=np.float32); self.vB2 = np.zeros_like(self.b2, dtype=np.float32)
            self.mB3 = np.zeros_like(self.b3, dtype=np.float32); self.vB3 = np.zeros_like(self.b3, dtype=np.float32)
            self.t_adam = 0

    def calcul_couche_sortie(self, state: NDArray[np.float32]) -> NDArray[np.float32]:
            A0 = np.array(state)
            Z1 = np.dot(self.w1, A0) + self.b1
            A1 = relu(Z1)
            Z2 = np.dot(self.w2, A1) + self.b2
            A2 = relu(Z2)
            Z3 = np.dot(self.w3, A2) + self.b3
            A3 = softmax(Z3)
            return A3

    def entrainement_reseau(self, samples: NDArray[np.float32]):
        for _ in range(NB_ENTRAINEMENT_BATCH):
            indices = np.random.choice(len(samples), TAILLE_BATCHS, replace=False)
            selection = samples[indices]

            # --- 1. Extraction des données du batch ---
            states_batch     = selection[:, :self.TAILLE_STATE]
            actions_batch    = selection[:, self.TAILLE_STATE * 2].astype(np.int64)
            prob_old_batch   = selection[:, self.TAILLE_STATE * 2 + 2]
            advantages       = selection[:, self.TAILLE_STATE * 2 + 4]  # colonne GAE

            # --- 2. Forward Pass ---
            Z1 = states_batch @ self.w1.T + self.b1
            A1 = relu(Z1)
            Z2 = A1 @ self.w2.T + self.b2
            A2 = relu(Z2)
            Z3 = A2 @ self.w3.T + self.b3
            A3 = softmax(Z3)

            prob_new = A3[np.arange(TAILLE_BATCHS), actions_batch]
            ratio = np.divide(prob_new, (prob_old_batch + 1e-8))

            one_hot = np.zeros((TAILLE_BATCHS, self.NB_ACTIONS_POSSIBLE), dtype=np.float32)
            one_hot[np.arange(TAILLE_BATCHS), actions_batch] = 1.0

            # normalisation par batch
            advantages = advantages - np.mean(advantages)
            advantages = advantages / (np.std(advantages) + 1e-8)

            mask_clip = (
                ((advantages > 0) & (ratio > 1 + EPSILON_CLIP_VALUE))
                |
                ((advantages < 0) & (ratio < 1 - EPSILON_CLIP_VALUE))
            )

            gradient_coeff = -ratio * advantages
            gradient_coeff[mask_clip] = 0.0
            gradientaC = gradient_coeff[:, None] * (one_hot - A3)

            log_A3 = np.log(A3 + 1e-8)
            entropie = -np.sum(A3 * log_A3, axis=1, keepdims=True)
            gradient_entropie = A3 * (log_A3 + entropie)
            gradientaC += coeff_entropie * gradient_entropie

            # --- 4. Backward Pass ---
            delta3 = gradientaC

            delta2 = (delta3 @ self.w3) * relu_derivative(Z2)
            delta1 = (delta2 @ self.w2) * relu_derivative(Z1)

            # Gradients pour W1, W2, B1, B2
            dW1 = (delta1.T @ states_batch / TAILLE_BATCHS).astype(np.float32, copy=False)
            dW2 = (delta2.T @ A1 / TAILLE_BATCHS).astype(np.float32, copy=False)
            dW3 = (delta3.T @ A2 / TAILLE_BATCHS).astype(np.float32, copy=False)
            dB1 = np.sum(delta1, axis=0).astype(np.float32, copy=False) / TAILLE_BATCHS
            dB2 = np.sum(delta2, axis=0).astype(np.float32, copy=False) / TAILLE_BATCHS
            dB3 = np.sum(delta3, axis=0).astype(np.float32, copy=False) / TAILLE_BATCHS
            norme = np.sqrt(
                np.sum(dW1**2) +
                np.sum(dW2**2) +
                np.sum(dW3**2) +
                np.sum(dB1**2) +
                np.sum(dB2**2) +
                np.sum(dB3**2)
            )
            if norme > MAX_NORME_GRADIENT:
                facteur = MAX_NORME_GRADIENT / norme
                dW1 *= facteur
                dW2 *= facteur
                dW3 *= facteur
                dB1 *= facteur
                dB2 *= facteur
                dB3 *= facteur

            '''if self.t_adam % 1000 == 0:
                print()
                print(
                f"Vcurr [{np.min(V_curr):.3f}, {np.mean(V_curr):.3f}, {np.max(V_curr):.3f}] | "
                f"Vnext [{np.min(V_next):.3f}, {np.mean(V_next):.3f}, {np.max(V_next):.3f}] | "
                f"Grad {norme:.3f}")
                print(
                f"Reward [{np.min(rewards_batch):.3f}, "
                f"{np.mean(rewards_batch):.3f}, "
                f"{np.max(rewards_batch):.3f}]"
                )
                print()'''

            # --- 4. Mise à jour des poids ---
            self.t_adam += 1
            adam_update(self.w1, dW1, self.mW1, self.vW1, self.t_adam, learning_rate)
            adam_update(self.w2, dW2, self.mW2, self.vW2, self.t_adam, learning_rate)
            adam_update(self.w3, dW3, self.mW3, self.vW3, self.t_adam, learning_rate)
            adam_update(self.b1, dB1, self.mB1, self.vB1, self.t_adam, learning_rate)
            adam_update(self.b2, dB2, self.mB2, self.vB2, self.t_adam, learning_rate)
            adam_update(self.b3, dB3, self.mB3, self.vB3, self.t_adam, learning_rate)

    def export_reseau(self):
        np.savez(self.nom_fichier, W1=self.w1, W2=self.w2, W3=self.w3, B1=self.b1, B2=self.b2, B3=self.b3, mW1=self.mW1, mW2=self.mW2, mW3=self.mW3, mB1=self.mB1, mB2=self.mB2, mB3=self.mB3, vW1=self.vW1, vW2=self.vW2, vW3=self.vW3, vB1=self.vB1, vB2=self.vB2, vB3=self.vB3, t_adam=self.t_adam)
        print(f"Poids, biais exportés dans {self.nom_fichier}")
