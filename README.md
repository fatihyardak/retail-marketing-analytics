# Retail Marketing Analytics

Bu proje, perakende pazarlama analizi için geliştirilmiştir.

## Sanal Ortam Kurulumu

### 1. Sanal Ortamı Etkinleştirme

Windows PowerShell'de:
```powershell
.\venv\Scripts\Activate.ps1
```

Veya Windows Command Prompt'ta:
```cmd
venv\Scripts\activate.bat
```

Veya batch dosyasını çalıştırarak:
```cmd
activate_venv.bat
```

### 2. Paket Kurulumu

Sanal ortam etkinleştirildikten sonra:
```bash
pip install -r requirements.txt
```

### 3. Sanal Ortamdan Çıkış

```bash
deactivate
```

## Proje Yapısı

- `venv/` - Python sanal ortamı
- `requirements.txt` - Gerekli Python paketleri
- `activate_venv.bat` - Windows için sanal ortam etkinleştirme dosyası

## Kullanım

1. Sanal ortamı etkinleştirin
2. Gerekli paketleri yükleyin
3. Jupyter notebook'u başlatın: `jupyter notebook`
4. Analizlerinizi gerçekleştirin

## Not

Sanal ortam etkinleştirildiğinde, komut satırında `(venv)` öneki görünecektir.
