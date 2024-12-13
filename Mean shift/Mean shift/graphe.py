import matplotlib.pyplot as plt
import json

# Charger les données enregistrées
with open('tracking_positions.json', 'r') as f:
    positions = json.load(f)

with open('tracking_similarities.json', 'r') as f:
    similarities = json.load(f)

# Trajectoire de la fenêtre de suivi
positions_x, positions_y = zip(*positions)
plt.figure(figsize=(10, 6))
plt.plot(positions_x, positions_y, marker='o')
plt.title('Trajectoire de la fenêtre de suivi')
plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.grid(True)
plt.show()

# Similarité au fil du temps
plt.figure(figsize=(10, 6))
plt.plot(similarities, marker='o')
plt.title('Évolution de la similarité (Coefficient de Bhattacharyya)')
plt.xlabel('Itération')
plt.ylabel('Similarité')
plt.grid(True)
plt.show()
