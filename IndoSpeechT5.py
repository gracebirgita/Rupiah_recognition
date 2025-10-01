import torch
import soundfile as sf
import re
import os
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier

# Text normalization and cleaning functions
replacements = [
    ("â", "a"), ("á", "a"), ("à", "a"), ("é", "e"), ("è", "e"), 
    ("ê", "e"), ("î", "i"), ("ô", "o"), ("û", "u"), ("ç", "s"),
    ("ñ", "ny"), ("ł", "l"), ("ń", "n"), ("²", "2")
]

number_words = {
    0: "nol", 1: "satu", 2: "dua", 3: "tiga", 4: "empat", 5: "lima",
    6: "enam", 7: "tujuh", 8: "delapan", 9: "sembilan", 10: "sepuluh",
    11: "sebelas", 12: "dua belas", 13: "tiga belas", 14: "empat belas",
    15: "lima belas", 16: "enam belas", 17: "tujuh belas", 
    18: "delapan belas", 19: "sembilan belas", 20: "dua puluh",
    30: "tiga puluh", 40: "empat puluh", 50: "lima puluh", 
    60: "enam puluh", 70: "tujuh puluh", 80: "delapan puluh",
    90: "sembilan puluh", 100: "seratus", 1000: "seribu"
}

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\']', '', text)
    return ' '.join(text.split())

def number_to_words(number):
    if number < 20:
        return number_words[number]
    elif number < 100:
        tens, unit = divmod(number, 10)
        return number_words[tens * 10] + (" " + number_words[unit] if unit else "")
    elif number < 1000:
        hundreds, remainder = divmod(number, 100)
        return ("seratus" if hundreds == 1 else number_words[hundreds] + " ratus") + (" " + number_to_words(remainder) if remainder else "")
    elif number < 1000000:
        thousands, remainder = divmod(number, 1000)
        return ("seribu" if thousands == 1 else number_to_words(thousands) + " ribu") + (" " + number_to_words(remainder) if remainder else "")
    elif number < 1000000000:
        millions, remainder = divmod(number, 1000000)
        return number_to_words(millions) + " juta" + (" " + number_to_words(remainder) if remainder else "")
    else:
        return str(number)

def replace_numbers_with_words(text):
    def replace(match):
        number = int(match.group())
        return number_to_words(number)
    return re.sub(r'\b\d+\b', replace, text)

def cleanup_text(text):
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text

# Speaker embedding functions
def load_speaker_model():
    spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return EncoderClassifier.from_hparams(
        source=spk_model_name,
        run_opts={"device": device},
        savedir=os.path.join("/tmp", spk_model_name)
    )

def create_speaker_embedding_from_wav(wav_path, speaker_model):
    waveform, sample_rate = sf.read(wav_path)
    if sample_rate != 16000:
        from librosa import resample
        waveform = resample(waveform, orig_sr=sample_rate, target_sr=16000)
    
    waveform = torch.FloatTensor(waveform).unsqueeze(0).to(speaker_model.device)
    
    with torch.no_grad():
        conv_layer = speaker_model.mods.embedding_model.blocks[0].conv
        conv_layer.padding = (1)
        speaker_embeddings = speaker_model.encode_batch(waveform)
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        return speaker_embeddings.squeeze().cpu().numpy()

def load_tts_components():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load models
    processor = SpeechT5Processor.from_pretrained("masp307/speecht5_finetuned_indotts")
    model = SpeechT5ForTextToSpeech.from_pretrained("masp307/speecht5_finetuned_indotts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
    # Load a sample speaker embedding
    speaker_embedding = torch.load("speakers\speaker2.pt")
    
    return processor, model, vocoder, speaker_embedding

# Main TTS function
def text_to_speech(text, output_file="output.wav"):
    processor, model, vocoder, speaker_embedding = load_tts_components()
    
    # Text processing pipeline
    converted_text = replace_numbers_with_words(text)
    cleaned_text = cleanup_text(converted_text)
    final_text = normalize_text(cleaned_text)
    
    # Generate speech
    inputs = processor(text=final_text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)
    
    # Save to file
    os.makedirs("tts_output", exist_ok=True)
    sf.write(output_file, speech.numpy(), samplerate=16000)
    return output_file