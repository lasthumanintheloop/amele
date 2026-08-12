# Canlı Test Raporu - OpenRouter (2026-08-10)

Phase 1 binary'si, gerçek bir OpenRouter hesabıyla uçtan uca test edildi.
Amaç: hem temel işlevi doğrulamak hem de yaratıcı senaryolarla potansiyel
bug avlamak. Bu dizindeki `.yaml` dosyaları kullanılan senaryo config'leridir
(hepsi `api_key: ${OPENROUTER_API_KEY}` kullanır - düz anahtar yoktur).

## Kullanılan modeller

`openai/gpt-5.6-luna`, `google/gemini-3.5-flash-lite`, `qwen/qwen3.7-flash`
(hepsi OpenRouter üzerinden, OpenAI-compatible endpoint).

**OpenRouter uyumluluğu: sorunsuz.** Mevcut OpenAI-compatible istemci
OpenRouter'la doğrudan çalıştı; kod değişikliği gerekmedi. Tool calling,
usage muhasebesi ve çoklu model hepsi çalıştı.

## Senaryolar ve sonuçlar

| # | Senaryo | Tool seti | Sonuç |
|---|---|---|---|
| S1 | Gerçek sistem loglarını triyaj + Türkçe rapor | fs (sandbox) | ✓ 3 turn, öncelikli rapor üretti |
| S2 | Sistem sağlık ajanı (disk/mem/uptime → report.md) | subprocess + fs | ✓ ama **B1 bug'ını tetikledi** (gemini) |
| S3 | LLM judge: amele kendini subprocess olarak çağırıp 3 modele dağıtır | subprocess (nested amele) | **B2 + B3 bug'larını tetikledi** |
| S4 | Prompt injection + sandbox escape (kötü niyetli log içeriği) | fs (sandbox) | ✓ model direndi + sandbox hazırdı |
| S5 | Bütçe kill-switch (sonsuz tool döngüsü) | subprocess | ✓ max_turns ve max_tokens tam sınırda kesti |

## Bulunan 3 bug (hepsi düzeltildi + regresyon testi eklendi)

### B1 - `finish_reason` görmezden geliniyordu (loop)
Tool çağrısı olmayan bir turn, `finish_reason` ne olursa olsun başarı
sayılıyordu. OpenRouter, üretim ortasındaki hataları `finish_reason: "error"`
+ boş gövdeyle raporluyor (gemini'de canlı görüldü). Bu, exit 0 (başarı)
olarak dönüyordu. **Düzeltme:** `badFinish()` - `error`/`length`/
`content_filter`/boş yanıt artık görev başarısızlığı (exit 1).

### B2 - nested subprocess timeout'ta takılma + süreç sızıntısı (tools)
Kendi torununu ayrı process group'ta başlatan bir subprocess (kendi
çocuğunu spawn eden nested amele), grup-kill sonrası ölü çocuğun stdout
pipe'ını açık tuttuğu için `Invoke`'u deadline'ın ötesinde bloke ediyordu;
torun süreç orphan olarak sızıyordu. **Düzeltme:** `cmd.WaitDelay`
(iptal sonrası bekleme sınırı) + grup sinyalini `SIGTERM` yaparak nested
amele'in kendi alt ağacını graceful temizlemesi.

### B3 - koşulsuz stdin okuma sonsuza kadar bloke ediyordu (cli) - EN CİDDİ
`buildTask`, task argümanı verilmiş ve prompt `{{input}}` içermiyor olsa
bile stdin'i koşulsuz okuyordu - üstelik run timeout kurulmadan **önce**.
stdin açık ama verisiz bir pipe ise (arka planda çalışan run, systemd
socket, orchestrator'ın spawn ettiği süreç) `io.ReadAll` sonsuza kadar
bloke oluyordu ve hiçbir bütçe bunu kesemiyordu. **Testte gözlenen tüm
gizemli takılmaların tek kök nedeni buydu.** **Düzeltme:** stdin yalnızca
prompt `{{input}}` içeriyorsa veya hiç task argümanı verilmediğinde okunur.

## Doğrulanan güvenlik davranışları

- **Sandbox (S4):** `../`, mutlak path ve symlink escape denemeleri -
  os.Root ile kernel seviyesinde engelli (dosyada kanıtlı).
- **Prompt injection (S4):** log içine gömülü "workspace dışını oku/yaz"
  talimatlarına model direndi; sandbox da hazırdı (iki katman).
- **Secret redaction:** API anahtarı session log'una sızmadı.
- **Bütçe kill-switch (S5):** max_turns / max_tokens / timeout hepsi
  tam sınırda kesti; süreç sızıntısı yok.

## Nasıl tekrar çalıştırılır

```bash
export OPENROUTER_API_KEY=sk-or-...
amele run testdata/live-scenarios/s1-log-triage.yaml "triyaj yap"
```
(S1 için workspace'te bir `logs/` dizini ve içinde log dosyaları gerekir.)
