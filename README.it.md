<div align="center">

# ComfyUI Upscaler TensorRT ⚡

[![python](https://img.shields.io/badge/python-3.12.3-green)](https://www.python.org/downloads/release/python-3123//)
[![cuda](https://img.shields.io/badge/cuda-13.1-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.14.1.48-green)](https://developer.nvidia.com/tensorrt)
[![by-nc-sa/4.0](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

</div>

Questo progetto fornisce un'implementazione [Tensorrt](https://github.com/NVIDIA/TensorRT) per un rapido upscaling di immagini utilizzando modelli all'interno di ComfyUI (2-4x più veloce)

**Ultimo test**: 12 Gennaio 2026 (ComfyUI v0.8.2@c623804 | Torch 2.9.1 | Tensorrt 10.14.1.48 | Python 3.12.3 | RTX5090 | CUDA 13.1 | Ubuntu 24.04)

<p align="center">
  <img src="assets/node_v3.png" style="height: 400px" />
</p>

## ⭐ Supporto
Se ti piacciono i miei progetti e desideri vedere aggiornamenti e nuove funzionalità, considera di supportarmi. Aiuta molto!

[![ComfyUI-QwenVL-Mod](https://img.shields.io/badge/ComfyUI--QwenVL--Mod-blue?style=flat-square)](https://github.com/huchukato/ComfyUI-QwenVL-Mod)
[![comfy-tagcomplete](https://img.shields.io/badge/comfy--tagcomplete-blue?style=flat-square)](https://github.com/huchukato/comfy-tagcomplete)
[![ComfyUI-HuggingFace](https://img.shields.io/badge/ComfyUI--HuggingFace-blue?style=flat-square)](https://github.com/huchukato/ComfyUI-HuggingFace)
[![ComfyUI-Rife-Tensorrt](https://img.shields.io/badge/ComfyUI--Rife--Tensorrt-blue?style=flat-square)](https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto)
[![stemify-audio-splitter](https://img.shields.io/badge/stemify--audio--splitter-blue?style=flat-square)](https://github.com/huchukato/stemify-audio-splitter)

[![ComfyUI-Whisper](https://img.shields.io/badge/ComfyUI--Whisper-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Whisper)
[![ComfyUI_InvSR](https://img.shields.io/badge/ComfyUI__InvSR-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI_InvSR)
[![ComfyUI-Thera](https://img.shields.io/badge/ComfyUI--Thera-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Thera)
[![ComfyUI-Video-Depth-Anything](https://img.shields.io/badge/ComfyUI--Video--Depth--Anything-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything)
[![ComfyUI-PiperTTS](https://img.shields.io/badge/ComfyUI--PiperTTS-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-PiperTTS)

[![buy-me-coffees](https://i.imgur.com/3MDbAtw.png)](https://buymeacoffee.com/huchukato)
[![paypal-donation](https://i.imgur.com/w5jjubk.png)](https://paypal.me/yuvraj108c)
---

## ⏱️ Performance

_Nota: I seguenti risultati sono stati benchmarkati su engine FP16 all'interno di ComfyUI, utilizzando 100 frame identici_

| Dispositivo |     Modello     | Risoluzione Input (WxH) | Risoluzione Output (WxH) | FPS |
| :---------: | :-------------: | :--------------------: | :---------------------: | :-: |
|   RTX5090   | 4x-UltraSharp  |       512 x 512        |       2048 x 2048       | 12.7 |
|   RTX5090   | 4x-UltraSharp  |      1280 x 1280       |       5120 x 5120       | 2.0  |
|   RTX4090   | 4x-UltraSharp  |       512 x 512        |       2048 x 2048       | 6.7  |
|   RTX4090   | 4x-UltraSharp  |      1280 x 1280       |       5120 x 5120       | 1.1  |
|   RTX3060   | 4x-UltraSharp  |       512 x 512        |       2048 x 2048       | 2.2  |
|   RTX3060   | 4x-UltraSharp  |      1280 x 1280       |       5120 x 5120       | 0.35 |

## 🚀 Installazione

### 🎯 Installazione Completamente Automatica (Consigliata)
Questo nodo presenta un rilevamento completamente automatico di CUDA e l'installazione di TensorRT!
Quando ComfyUI carica il nodo per la prima volta, eseguirà:
1. Rilevamento automatico della tua versione CUDA (12 o 13)
2. Installazione automatica dei pacchetti TensorRT appropriati
3. Configurazione di tutto per un funzionamento senza interruzioni
Nessun passaggio manuale richiesto! Basta clonare il repo e riavviare ComfyUI.

### 📦 Opzioni di Installazione Manuale
Se preferisci l'installazione manuale o riscontri problemi:
Script di auto-installazione:
```bash
# Linux/macOS
./install.sh

# Windows
install.bat

# Python (cross-platform)
python install.py
```

File requirements manuali:
```bash
# Per CUDA 13 (serie RTX 50)
pip install -r requirements.txt

# Per CUDA 12 (serie RTX 30/40) - METODO LEGACY
pip install -r requirements_cu12.txt
```

💡 Nota: Il file requirements_cu12.txt è fornito come metodo legacy. L'installazione automatica è fortemente raccomandata in quanto gestisce il rilevamento CUDA e l'installazione dei pacchetti in modo trasparente.

### 📦 CUDA Toolkit Richiesto
Il nodo rileva automaticamente la tua installazione CUDA tramite le variabili d'ambiente CUDA_PATH o CUDA_HOME.

```
CUDA_PATH
```

```
CUDA_HOME
```

Se CUDA non viene rilevato, scarica da: [https://developer.nvidia.com/cuda-13-0-2-download-archive](https://developer.nvidia.com/cuda-13-0-2-download-archive)

## 🛠️ Modelli Supportati

- Questi modelli upscaler sono stati testati per funzionare con Tensorrt. I modelli Onnx sono disponibili [qui](https://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/tree/main)
- I modelli tensorrt esportati supportano risoluzioni dinamiche delle immagini da 256x256 a 1280x1280 px (es 960x540, 512x512, 1280x720 etc).

   - [4x-AnimeSharp](https://openmodeldb.info/models/4x-AnimeSharp)
   - [4x-UltraSharp](https://openmodeldb.info/models/4x-UltraSharp)
   - [4x-WTP-UDS-Esrgan](https://openmodeldb.info/models/4x-WTP-UDS-Esrgan)
   - [4x_NMKD-Siax_200k](https://openmodeldb.info/models/4x-NMKD-Siax-CX)
   - [4x_RealisticRescaler_100000_G](https://openmodeldb.info/models/4x-RealisticRescaler)
   - [4x_foolhardy_Remacri](https://openmodeldb.info/models/4x-Remacri)
   - [RealESRGAN_x4](https://openmodeldb.info/models/4x-realesrgan-x4plus)
   - [4xNomos2_otf_esrgan](https://openmodeldb.info/models/4x-Nomos2-otf-esrgan)
   - [4x-ClearRealityV1](https://openmodeldb.info/models/4x-ClearRealityV1)
   - [4x_UniversalUpscalerV2-Neutral_115000_swaG](https://openmodeldb.info/models/4x-UniversalUpscalerV2-Neutral)
   - [4x-UltraSharpV2_Lite](https://huggingface.co/Kim2091/UltraSharpV2) 

## ☀️ Utilizzo

- Carica il [workflow di esempio](assets/tensorrt_upscaling_workflow.json) 
- Scegli il modello appropriato dal menu a tendina
- L'engine tensorrt verrà costruito automaticamente
- Carica un'immagine con risoluzione tra 256-1280px
- Imposta `resize_to` per ridimensionare le immagini upscaled a risoluzioni fisse o personalizzate

## 🔧 Modelli Personalizzati
- Per esportare altri modelli ESRGAN, dovrai prima costruire il modello onnx, usando [export_onnx.py](scripts/export_onnx.py) 
- Posiziona il modello onnx in `/ComfyUI/models/onnx/TUO_MODELLO.onnx`
- Poi, aggiungi il tuo modello a questa lista [load_upscaler_config.json](load_upscaler_config.json)
- Infine, esegui lo stesso workflow e scegli il tuo modello
- Se hai testato un altro modello tensorrt funzionante, fammelo sapere per aggiungerlo ufficialmente a questo nodo

## 🚨 Aggiornamenti
### 12 Gennaio 2026
- Aggiunti più fattori di scala di ridimensionamento
- Aggiunto ridimensionamento a risoluzione personalizzata

### 27 Agosto 2025
- Supporto per 4x-UltraSharpV2_Lite, 4x_UniversalUpscalerV2-Neutral_115000_swaG, 4x-ClearRealityV1
- Caricamento modelli da config [PR#57](https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt/pull/57)

### 30 Aprile 2025
- Merge https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt/pull/48 di @BiiirdPrograms per correggere il soft-lock sollevando un errore quando le dimensioni dell'immagine di input non sono supportate
### 4 Marzo 2025 (breaking)
- Gli engine tensorrt automatici vengono costruiti dal workflow stesso, per semplificare il processo per persone non tecniche
- Separazione del caricamento modelli e elaborazione tensorrt in nodi diversi
- Ottimizzazione del post-processing
- Aggiornamento script di esportazione onnx

## ⚠️ Problemi noti

- Se aggiorni la versione di tensorrt, dovrai ricostruire gli engine
- Attualmente funzionano solo modelli con architettura ESRGAN
- Alto utilizzo di RAM durante l'esportazione da `.pth` a `.onnx`

## 🤖 Ambiente testato

- Ubuntu 24.04, Debian 12
- Windows 11

## 👏 Crediti

- [NVIDIA/Stable-Diffusion-WebUI-TensorRT](https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT)
- [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## Licenza

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
