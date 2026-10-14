import numpy as np
import matplotlib.pyplot as plt

B = 20.0         # fondo di conteggi 
I0 = 400.0      # picco 
alpha = 0.90    # larghezza inviluppo mm^1
delta = 5.20    # quanto sono fitte le oscillazioni mm^-1


x = np.linspace(-5.0, 5.0, 161)  #in mm


E = np.sinc(alpha * x / np.pi) ** 2 
C = np.cos(delta * x) ** 2

mu = B + I0 * E * C


#test
print("Verifiche sul modello mu(x):")

print("Finita ovunque?") 
print(np.all(np.isfinite(mu)))

print("non negativa ovunque?")
print(np.all(mu >= 0))

print("simmetrica?")
print(np.allclose(mu, mu[::-1]))

idx0 = np.argmin(np.abs(x))  # indice del punto più vicino a x=0
print("mu(0) caolcolato vs B + I0 atteso:")
print("calcolato")
#print(mu[idx0]:.6f)
print("atteso")
#print(B + I0:.6f)

print(np.isclose(mu[idx0], B + I0))

#grafico
plt.figure(figsize=(9, 5))
plt.plot(x, mu, label="mu(x)", color="tab:blue")
plt.axhline(B, color="tab:gray", label=r"Fondo $B$")
plt.plot(x, B + I0 * E, color="tab:red", label=r"Inviluppo superiore $B + I_0 E(x)$")

plt.xlabel("x [mm]")
plt.ylabel("conteggi per bin")
plt.title("Modello fisico: conteggi medi attesi")
plt.legend()
plt.tight_layout()
plt.show()

#risposte alle domande del testo:
# 1: alpha: più alpha è grande, più stretto è il lobo centrale dell'inviluppo
#
# 2: un delta più grande significa frange piu' fitte.
#
# 3: il limite per x->0 vale 1, quindi è singolarità eliminabile

#############
## parte 2 ##
#############


#rng per errore statistico nella misura
rng = np.random.default_rng(seed=1)
n = rng.poisson(mu) 

errori = np.sqrt(n)  # errore statistico sulla misura 
errori = np.where(errori == 0, 1, errori)  # se n=0, errore=1 

#quali punti hanno !x!<1.2mm
regione = np.abs(x) < 1.2

N_c = np.sum(n[regione])  #somma dei conteggi 
mu_c = np.sum(mu[regione])  #somma dei valori attesi

scarto = N_c - mu_c
atteso = np.sqrt(mu_c)
scarto_rel = scarto / atteso

print("N_c - mu_c = ", scarto)
print("atteso = ", atteso)
print("scarto relativo = ", scarto_rel)

#grafico 
plt.figure(figsize=(9, 5))
plt.errorbar(x, n, yerr=errori, fmt="o", color="tab:blue",
             ecolor="tab:blue", elinewidth=1, capsize=2)
plt.plot(x, mu, color="tab:red")
plt.xlabel("x [mm]")
plt.ylabel("conteggi per bin")
plt.title("Simulazione di un'acquisizione (Poisson)")
plt.legend()
plt.show()


#############
## parte 3 ##
#############

M = 2000 #numero di esperimenti

n_matrix = rng.poisson(mu, size=(M, len(x)))   # shape: (2000, 161)

# Per ogni riga (esperimento), somma dei conteggi nella regione centrale

N_c_dist = np.sum(n_matrix[:, regione], axis=1)  # shape: (2000,)
#statistiche della distribuzione di N_c
media_Nc = np.mean(N_c_dist)
std_Nc = np.std(N_c_dist, ddof=1)          # deviazione standard, dof=1
err_media_Nc = std_Nc / np.sqrt(M)         # errore sulla media delle simulazioni
scarto_relativo = (media_Nc - mu_c) / mu_c  # scarto relativo media simulata vs mu_c
 
sigma_teorica = np.sqrt(mu_c)              # sigma attesa da Poisson: sqrt(mu_c)
 
print("\nDistribuzione dello stimatore N_c su M =", M, "pseudo-esperimenti:")
print(f"media simulata <N_c>:          {media_Nc:.2f}")
print(f"deviazione standard (dof=1):   {std_Nc:.2f}")
print(f"errore sulla media simulata:   {err_media_Nc:.3f}")
print(f"scarto relativo (media-mu_c)/mu_c: {scarto_relativo:.4%}")
print(f"sigma teorica sqrt(mu_c):      {sigma_teorica:.2f}")
print(f"confronto std simulata vs teorica: "
      f"{std_Nc:.2f} contro {sigma_teorica:.2f} "
      f"(differenza relativa {(std_Nc - sigma_teorica) / sigma_teorica:.2%})")
 
# Istogramma della distribuzione di N_c
plt.figure(figsize=(9, 5))
plt.hist(N_c_dist, bins=40, color="tab:blue", alpha=0.7,
         edgecolor="black", label=r"Distribuzione simulata di $N_c$")
plt.axvline(media_Nc, color="tab:orange", linestyle="-",
            linewidth=2, label=rf"Media simulata $\langle N_c \rangle$ = {media_Nc:.1f}")
plt.axvline(mu_c, color="tab:red", linestyle="--",
            linewidth=2, label=rf"Valore atteso $\mu_c$ = {mu_c:.1f}")
 
plt.xlabel(r"$N_c$ (conteggi nella regione centrale)")
plt.ylabel("frequenza")
plt.title(f"Distribuzione dello stimatore $N_c$ su {M} pseudo-esperimenti")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

