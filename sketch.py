import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import sounddevice as sd
import threading
import queue
import time

# ============================
# CONFIGURAÇÕES
# ============================
W = 500                     # tamanho do canvas
GRID_STEP = 50
I_MIN, I_MAX, I_STEP = 0, 2, 0.1
CLAMP = 20
AMPLITUDE = 40
DAMPING = 0.98
MASS_SCALE = 0.5
REPULSION_RADIUS = 30
WALL_REPULSION = 0.3
FORCE_STRENGTH = 0.01

# Parâmetros de sonificação
SAMPLE_RATE = 44100
AUDIO_DURATION = 0.15       # duração de cada nota (s)
NOTE_QUEUE = queue.Queue()  # fila para enviar notas ao thread de áudio

# ============================
# CLASSE PARTÍCULA (CÉLULA COM ESTADO PERSISTENTE)
# ============================
class Particle:
    def __init__(self, x, y, i):
        self.x = x
        self.y = y
        self.i = i
        self.cx = x
        self.cy = y
        self.vx = 0.0
        self.vy = 0.0

        # Estado persistente (evoluem com o tempo em vez de usar f global)
        self.phase = i  # Fase própria inicial
        self.energy = 1.0 # Energia inicial (axioma)

        self.radius = 10.0
        self.mass = 1.0
        self.hue = 0.0

    def update_state(self, dt=1.0):
        # A energia evolui baseada na sua própria dinâmica
        self.energy += 0.01 * np.sin(self.phase) * dt
        self.energy = np.clip(self.energy, 0.5, 2.0)

        # A fase própria evolui baseada na energia, sem usar f global
        self.phase += 0.05 * self.energy * dt

        # O ruído/perturbação depende da fase própria
        n = (0.5 + 0.5 * np.sin(self.x/30 * 1.3 + self.phase)
             * np.cos(self.y/30 * 1.7)
             * np.sin(self.phase * 2.1))

        # Raio e massa são funções do estado interno
        self.radius = 20 * max(0.1, abs(n)) * self.energy
        self.mass = self.radius * MASS_SCALE + 0.1

        # A cor (hue) depende da fase própria e posição
        I = self.phase + self.x*2 + self.y
        self.hue = ((I * 50) % 360) / 360

    def phase_force(self):
        # Potencial do campo de fase baseado na fase própria
        I = self.phase + self.x*2 + self.y
        target_x = self.x + CLAMP * np.cos(I)
        target_y = self.y + CLAMP * np.sin(I)
        dx = target_x - self.cx
        dy = target_y - self.cy
        strength = FORCE_STRENGTH * 2.0
        return dx * strength, dy * strength

    def repulsion_from(self, other):
        dx = self.cx - other.cx
        dy = self.cy - other.cy
        dist = np.sqrt(dx*dx + dy*dy) + 1e-6
        min_dist = self.radius + other.radius

        # Colisão elástica / potencial de repulsão entre células
        if dist < min_dist:
            overlap = min_dist - dist

            # Troca de energia (interação do estado interno)
            transfer = 0.05 * (other.energy - self.energy)
            self.energy += transfer
            other.energy -= transfer

            force = overlap * 0.1 / (self.mass + 0.01)
            return force * dx / dist, force * dy / dist
        return 0.0, 0.0

    def apply_forces(self, neighbors):
        fx, fy = 0.0, 0.0

        # Força de fase
        pf_x, pf_y = self.phase_force()
        fx += pf_x; fy += pf_y

        # Interação com vizinhos
        for other in neighbors:
            rx, ry = self.repulsion_from(other)
            fx += rx; fy += ry

        # Amortecimento (inércia/viscosidade)
        fx -= self.vx * 0.05
        fy -= self.vy * 0.05

        # Potencial das paredes
        if self.cx < 5:   fx += WALL_REPULSION
        if self.cx > W-5: fx -= WALL_REPULSION
        if self.cy < 5:   fy += WALL_REPULSION
        if self.cy > W-5: fy -= WALL_REPULSION

        # Potencial de atração de volta à âncora original (clamp suave)
        dx = self.cx - self.x
        dy = self.cy - self.y
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > CLAMP:
            pull = (dist - CLAMP) * 0.1
            fx -= pull * dx / dist
            fy -= pull * dy / dist

        # F = m*a
        ax = fx / self.mass
        ay = fy / self.mass
        self.vx += ax
        self.vy += ay
        self.vx *= DAMPING
        self.vy *= DAMPING
        self.cx += self.vx
        self.cy += self.vy

# ============================
# INICIALIZAÇÃO
# ============================
particles = []
for x in range(0, 600, GRID_STEP):
    for y in range(0, 600, GRID_STEP):
        for i in np.arange(I_MIN, I_MAX, I_STEP):
            p = Particle(x, y, i)
            particles.append(p)

# ============================
# SONIFICAÇÃO
# ============================
def generate_tone(freq, amp, dur, timbre_hue, sample_rate=SAMPLE_RATE):
    """Gera uma onda com timbre dependendo do hue (mistura de senóide, quadrada e dente de serra)"""
    n = int(sample_rate * dur)
    t = np.linspace(0, dur, n, False)

    # Envelope de amplitude para evitar clicks
    envelope = np.sin(np.pi * t / dur)

    # Timbre mapeado a partir de hue (cor):
    # hue ~ 0.0 a 0.33 -> mistura de senoide e quadrada
    # hue ~ 0.33 a 0.66 -> mistura de quadrada e serra
    # hue ~ 0.66 a 1.0 -> mistura de serra e senoide
    sine = np.sin(2 * np.pi * freq * t)
    square = np.sign(np.sin(2 * np.pi * freq * t))
    saw = 2.0 * (freq * t - np.floor(freq * t + 0.5))

    if timbre_hue < 0.33:
        mix = timbre_hue / 0.33
        wave = (1 - mix) * sine + mix * square
    elif timbre_hue < 0.66:
        mix = (timbre_hue - 0.33) / 0.33
        wave = (1 - mix) * square + mix * saw
    else:
        mix = (timbre_hue - 0.66) / 0.34
        wave = (1 - mix) * saw + mix * sine

    # Normaliza a onda gerada e aplica amplitude/envelope
    wave = wave * amp * envelope
    return wave

def audio_worker():
    while True:
        try:
            freq, amp, dur, hue = NOTE_QUEUE.get(timeout=0.1)
            wave = generate_tone(freq, amp, dur, hue)
            sd.play(wave, SAMPLE_RATE, blocking=True)
        except queue.Empty:
            continue

audio_thread = threading.Thread(target=audio_worker, daemon=True)
audio_thread.start()

def sonify(particles, frame):
    if frame % 3 != 0:
        return
    # Amostrar aleatoriamente
    sample_count = 8
    idxs = np.random.choice(len(particles), min(sample_count, len(particles)), replace=False)
    for idx in idxs:
        p = particles[idx]

        # Mapeamento 1: Posição Y e X para Frequência Base
        freq_base = 200 + (p.cx / W) * 400 + (p.cy / W) * 200

        # Mapeamento 2: Raio (r) para Modulação de Frequência
        freq_mod = 1 + (p.radius / 20) * 0.5
        freq = freq_base / freq_mod

        # Mapeamento 3: Amplitude (amp) baseada na energia e proximidade do centro
        center_dist = np.hypot(p.cx - W/2, p.cy - W/2)
        vol = 0.1 + 0.2 * (1 - center_dist / (W/2))
        vol *= p.energy
        vol = np.clip(vol, 0.0, 1.0)

        # Duração mapeada à inércia (velocidade) e estado persistente (fase)
        speed = np.hypot(p.vx, p.vy)
        dur = 0.05 + 0.1 * min(speed, 1.0) + 0.02 * abs(np.sin(p.phase))

        # Mapeamento 4: Cor (hue) para Timbre
        NOTE_QUEUE.put((freq, vol, dur, p.hue))

# ============================
# ANIMAÇÃO
# ============================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('black')
ax.set_xlim(0, W)
ax.set_ylim(W, 0)
ax.set_aspect('equal')
ax.axis('off')

collection = PatchCollection([], facecolors=[], edgecolors='none', alpha=0.6)
ax.add_collection(collection)

def init():
    collection.set_paths([])
    collection.set_facecolors([])
    return collection,

def animate(frame):
    # Atualizar estado interno
    for p in particles:
        p.update_state()

    # Calcular forças
    for i, p in enumerate(particles):
        neighbors = []
        for j, q in enumerate(particles):
            if i == j: continue
            dist = np.hypot(p.cx - q.cx, p.cy - q.cy)
            if dist < REPULSION_RADIUS * 3:
                neighbors.append(q)
                if len(neighbors) >= 8:
                    break
        p.apply_forces(neighbors)

    # Atualizar gráficos
    circles = []
    colors = []
    for p in particles:
        rgb = mcolors.hsv_to_rgb([p.hue, 0.85, 0.6 + 0.3 * (p.radius / 20)])
        circles.append(Circle((p.cx, p.cy), radius=p.radius))
        colors.append(rgb)

    collection.set_paths(circles)
    collection.set_facecolors(colors)

    # Sonificar o estado
    sonify(particles, frame)

    return collection,

if __name__ == '__main__':
    ani = FuncAnimation(fig, animate, init_func=init,
                        frames=200, interval=30, blit=True)
    plt.show()
