from packages.build_state import build_state
from packages.reseau_neurones_principal import Reseau_neurones_principal
from packages.reseau_neurones_V_estimate import Reseau_neurones_V_estimate
from packages.parametres_reseau_principal import TAILLE_STATE, NB_PARTIES, PERIODE_EXPORTATION, PERIODE_ENTRAINEMENT, gamma
from packages.parametres_V_estimate import PERIODE_GEL_V_OLD
import random
import numpy as np
from numpy.typing import NDArray
import copy

TAILLE_SAMPLE = 2 * TAILLE_STATE + 4

def jouer_une_partie(reseau_neurones_principal: Reseau_neurones_principal) -> NDArray[np.float32]:
    """Joue une partie complète et alimente le réseau en samples."""

    ## initialisation de la partie
    raise NotImplementedError("initialisation pas implementé !!!")

    samples_1_partie = np.zeros((0, TAILLE_SAMPLE), dtype=np.float32)

    partie_en_cours = True
    state2 = build_state(TAILLE_STATE)
    while partie_en_cours:
        state1 = state2
        action, proba = choisir_action(reseau_neurones_principal, state1)  # action >= 0
        reward, partie_en_cours = executer_action(action)
        state2 = build_state(TAILLE_STATE)
        extra = np.array([action, reward, proba, not partie_en_cours], dtype=np.float32)
        sample = np.concatenate([state1, state2, extra]).astype(np.float32, copy=False)
        samples_1_partie = np.concatenate([samples_1_partie, sample[None, :]], axis=0)
    return samples_1_partie

def choisir_action(reseau_neurones_principal: Reseau_neurones_principal, state: NDArray[np.float32]) -> tuple[int, float]:
    A3 = reseau_neurones_principal.calcul_couche_sortie(state)
    action = np.random.choice(len(A3), p=A3)
    return action, A3[action]  # action : 0, 1, 2...

def executer_action(action: int) -> tuple[float, bool]:
    raise NotImplementedError("executer action pas implementé")

    return reward, partie_en_cours

def entrainer(nb_parties: int = NB_PARTIES):
    reseau_neurones_principal = Reseau_neurones_principal("reseau_neurones_principal.npz")
    reseau_neurones_V_estimate = Reseau_neurones_V_estimate("reseau_neurones_V_estimate.npz")
    samples_P = np.zeros((0, TAILLE_SAMPLE), dtype=np.float32)
    samples_V = np.zeros((0, TAILLE_STATE + 2), dtype=np.float32)
    reseau_neurones_V_estimate_old = copy.deepcopy(reseau_neurones_V_estimate)
    nb_entrainements = 0
    nb_entrainements_V = 0

    for partie in range(nb_parties):
        new_samples = jouer_une_partie(reseau_neurones_principal)
        samples_P = np.concatenate([samples_P, new_samples])
        states  = new_samples[:, :TAILLE_STATE]
        states2 = new_samples[:, TAILLE_STATE:TAILLE_STATE * 2]
        rewards  = new_samples[:, TAILLE_STATE * 2 + 1]
        terminal = new_samples[:, TAILLE_STATE * 2 + 3]

        V_old      = reseau_neurones_V_estimate_old.calcul_V_estimate(states)
        V_next_old = reseau_neurones_V_estimate_old.calcul_V_estimate(states2)
        targets    = rewards + gamma * (1.0 - terminal) * V_next_old

        new_samples_V = np.concatenate(
            [states, V_old[:, None], targets[:, None]],
            axis=1, dtype=np.float32
        )
        samples_V = np.concatenate([samples_V, new_samples_V], axis=0)

        if len(samples_P) >= PERIODE_ENTRAINEMENT:
            nb_entrainements += 1
            reseau_neurones_principal.entrainement_reseau(samples_P, reseau_neurones_V_estimate)
            samples_P = np.zeros((0, TAILLE_SAMPLE), dtype=np.float32)

            reseau_neurones_V_estimate.entrainement_reseau(samples_V)
            samples_V = np.zeros((0, TAILLE_STATE + 2), dtype=np.float32)

            nb_entrainements_V += 1
            if nb_entrainements_V >= PERIODE_GEL_V_OLD:
                reseau_neurones_V_estimate_old = copy.deepcopy(reseau_neurones_V_estimate)
                nb_entrainements_V = 0

        if nb_entrainements >= PERIODE_EXPORTATION:
            nb_entrainements = 0
            print(f"progression : {partie * 100 / nb_parties} %")
            reseau_neurones_principal.export_reseau()
            reseau_neurones_V_estimate.export_reseau()

    reseau_neurones_principal.export_reseau()
    reseau_neurones_V_estimate.export_reseau()


if __name__ == "__main__":
    entrainer()