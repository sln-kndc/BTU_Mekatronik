# -*- coding: utf-8 -*-

#Mobil Robotlar Vize Ödevi




#1) Engel haritasi
#2) Bos alanlari dikdortgen bolgelere ayirma
#3) Komşu bolgelerin gecis noktalarini (AP) bulma
#4) Dijkstra ile yol planlama
#5) Iki tekerlekli robotun hareketini simule etme
#6) Grafik ve sonuc dosyasi olusturma


import csv
import heapq
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



R = 0.10     # Tekerlek yaricapi [metre] 
L = 0.50     # Iki tekerlek arasindaki mesafe [metre] 
HUCRE = 0.40   # Haritadaki bir karenin gercek boyu [metre] - varsayim
DT = 0.05      # Simulasyon zaman adimi [saniye] - varsayim
HIZ = 0.45     # En yuksek dogrusal hiz [m/s] - varsayim
DONUS = 1.20   # En yuksek acisal hiz [rad/s] - varsayim

# Koordinatlar sol ust kose (0, 0).
BASLANGIC = (2, 17)
HEDEF = (27, 10)

os.makedirs('sonuclar', exist_ok=True)


# ============= 1. ENGEL HARITASI =============
harita = np.zeros((20, 30), dtype=int)

# 1 = engel, 0 = bos alan
harita[0, :] = 1
harita[-1, :] = 1
harita[:, 0] = 1
harita[:, -1] = 1

# Ic duvarlar. Duvarlardaki acik yerler gecistir.
harita[3:17, 10:12] = 1
harita[9:13, 10:12] = 0
harita[2:17, 20:22] = 1
harita[5:9, 20:22] = 0

# Diger sabit engeller
harita[13:16, 4:8] = 1
harita[3:6, 25:28] = 1
harita[15:18, 24:27] = 1

YUKSEKLIK, GENISLIK = harita.shape

# Guvenlik icin engellerin etrafinda 1 hucrelik tampon olusturulur.
# Bu kisim makaledeki ozel haritalama yaziliminin yerine yapilan
# basit bir muhendislik varsayimidir.
guvenli_harita = harita.copy()
for y in range(YUKSEKLIK):
    for x in range(GENISLIK):
        if harita[y, x] == 1:
            for yy in range(max(0, y - 1), min(YUKSEKLIK, y + 2)):
                for xx in range(max(0, x - 1), min(GENISLIK, x + 2)):
                    guvenli_harita[yy, xx] = 1


# ============= 2. BOS ALANI DIKDORTGENLERE AYIRMA =============
# bolge = (sol, ust, sag, alt); sag ve alt sinir dahil degildir.
bolgeler = []
etiket = np.full((YUKSEKLIK, GENISLIK), -1, dtype=int)

for y in range(YUKSEKLIK):
    for x in range(GENISLIK):
        if guvenli_harita[y, x] == 1 or etiket[y, x] != -1:
            continue

        # Once saga dogru bos alan genisligi bulunur.
        genislik = 0
        while (x + genislik < GENISLIK and
               guvenli_harita[y, x + genislik] == 0 and
               etiket[y, x + genislik] == -1):
            genislik += 1

        # Ayni genislik bos kaldikca asagiya uzatilir.
        yukseklik = 0
        while y + yukseklik < YUKSEKLIK:
            satir_engelsiz = np.all(guvenli_harita[y + yukseklik, x:x + genislik] == 0)
            satir_bos = np.all(etiket[y + yukseklik, x:x + genislik] == -1)
            if not (satir_engelsiz and satir_bos):
                break
            yukseklik += 1

        numara = len(bolgeler)
        bolgeler.append((x, y, x + genislik, y + yukseklik))
        etiket[y:y + yukseklik, x:x + genislik] = numara


# ============= 3. KOMSU BOLGELERI VE AP NOKTALARINI BULMA =============
# AP = iki komsu dikdortgenin ortak sinirinin orta noktasi.
# AP bilgisi: (bolge1, bolge2, x, y)
ap_listesi = []

for i in range(len(bolgeler)):
    a = bolgeler[i]
    for j in range(i + 1, len(bolgeler)):
        b = bolgeler[j]

        # Dikey ortak sinir (yan yana bolgeler)
        if a[2] == b[0] or b[2] == a[0]:
            ust = max(a[1], b[1])
            alt = min(a[3], b[3])
            if alt > ust:
                x_ap = a[2] if a[2] == b[0] else b[2]
                ap_listesi.append((i, j, float(x_ap), (ust + alt) / 2))

        # Yatay ortak sinir (alt alta bolgeler)
        elif a[3] == b[1] or b[3] == a[1]:
            sol = max(a[0], b[0])
            sag = min(a[2], b[2])
            if sag > sol:
                y_ap = a[3] if a[3] == b[1] else b[3]
                ap_listesi.append((i, j, (sol + sag) / 2, float(y_ap)))


def nokta_merkezi(hucre):
    """Hucresinin tam ortasini bulur (harita olceginde)."""
    return (hucre[0] + 0.5, hucre[1] + 0.5)


def metreye_cevir(nokta):
    """Haritada Y asagi gider; gercek koordinatlarda Y yukari gider."""
    return (nokta[0] * HUCRE, (YUKSEKLIK - nokta[1]) * HUCRE)


# ============= 4. DIJKSTRA ILE YOL BULMA =============
# Graf dugumleri: AP noktalarinin hepsi + baslangic + hedef.
if (guvenli_harita[BASLANGIC[1], BASLANGIC[0]] == 1 or
        guvenli_harita[HEDEF[1], HEDEF[0]] == 1):
    raise ValueError('Baslangic veya hedef, engelin ya da guvenlik tamponunun icinde.')

noktalar = [(a[2], a[3]) for a in ap_listesi]
noktalar += [nokta_merkezi(BASLANGIC), nokta_merkezi(HEDEF)]

bas_id = len(noktalar) - 2
hedef_id = len(noktalar) - 1
bas_bolge = int(etiket[BASLANGIC[1], BASLANGIC[0]])
hedef_bolge = int(etiket[HEDEF[1], HEDEF[0]])


graf = [[] for _ in noktalar]

for bolge in range(len(bolgeler)):
    bolgedeki_noktalar = []
    for k, ap in enumerate(ap_listesi):
        if bolge == ap[0] or bolge == ap[1]:
            bolgedeki_noktalar.append(k)
    if bolge == bas_bolge:
        bolgedeki_noktalar.append(bas_id)
    if bolge == hedef_bolge:
        bolgedeki_noktalar.append(hedef_id)

    for i in range(len(bolgedeki_noktalar)):
        for j in range(i + 1, len(bolgedeki_noktalar)):
            a = bolgedeki_noktalar[i]
            b = bolgedeki_noktalar[j]
            uzaklik = math.dist(noktalar[a], noktalar[b])
            graf[a].append((b, uzaklik))
            graf[b].append((a, uzaklik))

# Dijkstra: en dusuk toplam uzaklikli noktayi sirayla arar.
uzakliklar = [float('inf')] * len(noktalar)
onceki = [-1] * len(noktalar)
uzakliklar[bas_id] = 0
kuyruk = [(0, bas_id)]

while kuyruk:
    maliyet, suanki = heapq.heappop(kuyruk)
    if maliyet > uzakliklar[suanki]:
        continue
    if suanki == hedef_id:
        break
    for komsu, kenar_uzunlugu in graf[suanki]:
        yeni = maliyet + kenar_uzunlugu
        if yeni < uzakliklar[komsu]:
            uzakliklar[komsu] = yeni
            onceki[komsu] = suanki
            heapq.heappush(kuyruk, (yeni, komsu))

if math.isinf(uzakliklar[hedef_id]):
    raise RuntimeError('Bu haritada hedefe ulasilan bir yol yok.')

# Geriye dogru ilerleyip bulunan rotayi ters ceviriyoruz.
rota_id = []
u = hedef_id
while u != -1:
    rota_id.append(u)
    u = onceki[u]
rota_id.reverse()

rota = [metreye_cevir(noktalar[i]) for i in rota_id]


# ============= 5. IKI TEKERLEKLI ROBOTUN HAREKETI =============
x, y = rota[0]
theta = 0.0              # Baslangic bakis acisi [radyan]
siradaki_nokta = 1
kayit = []


def aci_duzelt(aci):
    """Acinin -pi ile +pi arasinda kalmasini saglar."""
    return (aci + math.pi) % (2 * math.pi) - math.pi


for adim in range(6000):
    zaman = adim * DT

    # Hedeflenen ara noktaya geldiysek sonraki noktaya gec.
    while siradaki_nokta < len(rota):
        mesafe = math.dist((x, y), rota[siradaki_nokta])
        if mesafe >= 0.003:  # AP sinirina tam yakinlas, erken donme
            break
        siradaki_nokta += 1

    if siradaki_nokta == len(rota):
        kayit.append([zaman, x, y, theta, 0, 0, 0, 0])
        break

    hedef_x, hedef_y = rota[siradaki_nokta]
    mesafe = math.dist((x, y), (hedef_x, hedef_y))
    istenen_aci = math.atan2(hedef_y - y, hedef_x - x)
    aci_hatasi = aci_duzelt(istenen_aci - theta)

    # Robot tekerlekli oldugu icin yanal kayamaz.
    # Once dogru yone doner, sonra ileri gider.
    omega = max(-DONUS, min(DONUS, 3 * aci_hatasi))
    if abs(aci_hatasi) > 0.04:
        v = 0.0
    else:
        v = min(HIZ, 1.5 * mesafe)

    # Govde hizi -> sag ve sol tekerlek acisal hizlari
    sag = (v + omega * L / 2) / R
    sol = (v - omega * L / 2) / R

    # Tekerlek hizlari -> gercek govde hizlari
    v_gercek = (R / 2) * (sag + sol)
    omega_gercek = (R / L) * (sag - sol)

    kayit.append([zaman, x, y, theta, sol, sag, v_gercek, omega_gercek])

    # Bir sonraki zaman adimindaki robot konumu
    theta_orta = theta + omega_gercek * DT / 2
    x += v_gercek * math.cos(theta_orta) * DT
    y += v_gercek * math.sin(theta_orta) * DT
    theta = aci_duzelt(theta + omega_gercek * DT)

    # Engelin icine girilmedi mi? (guvenlik tamponuna gore kontrol)
    sutun = int(x / HUCRE)
    satir = int(YUKSEKLIK - y / HUCRE)
    if (sutun < 0 or sutun >= GENISLIK or
            satir < 0 or satir >= YUKSEKLIK or
            guvenli_harita[satir, sutun] == 1):
        raise RuntimeError(
            f'Robot guvenli haritadan cikti: t={zaman:.2f}s, x={x:.2f}, y={y:.2f}.'
        )
else:
    raise RuntimeError('Robot belirlenen surede hedefe ulasamadi.')

kayit = np.array(kayit)
son_hata = math.dist((x, y), rota[-1])
rota_uzunlugu = sum(math.dist(rota[i], rota[i + 1]) for i in range(len(rota) - 1))
iz_uzunlugu = sum(math.dist(kayit[i, 1:3], kayit[i + 1, 1:3]) for i in range(len(kayit) - 1))


# ============= 6. SONUCLARI GOSTERME =============
fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(harita, cmap='Greys', origin='upper',
          extent=[0, GENISLIK * HUCRE, 0, YUKSEKLIK * HUCRE],
          vmin=0, vmax=1, interpolation='nearest')

# Dikdortgen bolgeleri renklendirmeden kenar cizgileriyle goster.
for sol, ust, sag, alt in bolgeler:
    ax.add_patch(Rectangle((sol * HUCRE, (YUKSEKLIK - alt) * HUCRE),
                           (sag - sol) * HUCRE, (alt - ust) * HUCRE,
                           fill=False, edgecolor='teal', linewidth=0.7, alpha=0.65))

# Kirmizi: planlanan rota. Mavi: robotun izledigi gercek rota.
ax.plot([p[0] for p in rota], [p[1] for p in rota],
        'o--', color='crimson', label='RC-Map planlanan yol', markersize=3)
ax.plot(kayit[:, 1], kayit[:, 2], '-', color='royalblue',
        label='Robotun izledigi yol', linewidth=2)
ax.scatter(*rota[0], c='green', s=110, label='Baslangic', zorder=5)
ax.scatter(*rota[-1], c='orange', s=110, label='Hedef', zorder=5)
ax.set(xlabel='x (m)', ylabel='y (m)', title='RC-Map + Iki Tekerlekli Mobil Robot')
ax.legend(loc='upper left')
ax.grid(alpha=0.2)
ax.set_aspect('equal')
fig.savefig('sonuclar/1_robot_yolu.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(kayit[:, 0], kayit[:, 4], label='Sol tekerlek (rad/s)')
ax.plot(kayit[:, 0], kayit[:, 5], label='Sag tekerlek (rad/s)')
ax.set(xlabel='Zaman (s)', ylabel='Tekerlek acisal hizi (rad/s)',
       title='Simulasyon boyunca tekerlek hizlari')
ax.legend()
ax.grid(alpha=0.3)
fig.savefig('sonuclar/2_tekerlek_hizlari.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# Excel ile de acilabilen veri dosyasi
with open('sonuclar/3_robot_verileri.csv', 'w', newline='', encoding='utf-8') as dosya:
    yaz = csv.writer(dosya)
    yaz.writerow(['zaman_s', 'x_m', 'y_m', 'theta_rad',
                  'sol_rad_s', 'sag_rad_s', 'v_m_s', 'omega_rad_s'])
    yaz.writerows(kayit)

bilgi = (
    f'RC-Map bolge sayisi: {len(bolgeler)}\n'
    f'AP sayisi: {len(ap_listesi)}\n'
    f'Planlanan yol: {rota_uzunlugu:.3f} m\n'
    f'Robotun izledigi yol: {iz_uzunlugu:.3f} m\n'
    f'Hareket suresi: {kayit[-1, 0]:.2f} s\n'
    f'Hedef konum hatasi: {son_hata:.3f} m\n'
    'Robot engelden gecmedi (guvenlik haritasi kontrol edildi).\n'
)
print('\nSIMULASYON BITTI\n' + bilgi)
with open('sonuclar/4_sonuc_ozeti.txt', 'w', encoding='utf-8') as dosya:
    dosya.write(bilgi)
