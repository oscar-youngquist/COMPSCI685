from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import csv
from pathlib import Path

en_fr_name = "Helsinki-NLP/opus-mt-en-fr"
fr_en_name = "Helsinki-NLP/opus-mt-fr-en"

en_fr_model = MarianMTModel.from_pretrained(en_fr_name).cuda()
en_tokenizer = MarianTokenizer.from_pretrained(en_fr_name)

fr_en_model = MarianMTModel.from_pretrained(fr_en_name).cuda()
fr_tokenizer = MarianTokenizer.from_pretrained(fr_en_name)

k = 50
p = 0.8
temp = 0.1
generate_options = {
#     "do_sample": True,
#     "temperature": 0.8,
#     "top_p": 0.8,
#     "top_k": 50
    "num_beams": 10,
}

def translate_fr_en(toks):
    res = fr_en_model.generate(toks, **generate_options, num_return_sequences=2)
    txt = [en_tokenizer.decode(r, skip_special_tokens=True) for r in res]
    return txt

def translate_en_fr(txt):
    inputs = en_tokenizer(txt, return_tensors="pt", padding=True).to('cuda:0')
    toks = en_fr_model.generate(**inputs, **generate_options, num_return_sequences=5)
    return toks 

def paraphrase(txt):
    return [t.replace("▁", " ") for t in translate_fr_en(translate_en_fr(txt))]

if __name__ == "__main__":
    subsume_path = Path(__file__).absolute().parent.parent / "SubSumE_Data"
    path = Path(__file__).absolute().parent

    with open(subsume_path / 'processed_state_sentences.csv', 'r') as infile:
        fptrs = []
        for fname in [path / 'paraphrases' / f'paraphrase{i}.csv' for i in range(10)]:
            fptrs.append(open(fname, 'w'))

        try:
            reader = csv.DictReader(infile)
            writers = [csv.DictWriter(ptr, reader.fieldnames) for ptr in fptrs]
            [writer.writeheader() for writer in writers]

            with tqdm(total=22608) as progress:
                for row in reader:
                    paraphrases = []
                    try:
                        paraphrases = paraphrase(row['sentence'])
                    except Exception as e:
                        print(e)
                        paraphrases = 10*["<ERR>"]
                    for writer, p in zip(writers, paraphrases):
                        try:
                            row['sentence'] = p
                            writer.writerow(row)
                        except Exception as e:
                            print(f"Failed to write sentence {row['sid']}")
                            row['sentence'] = '<ERR>'
                            writer.writerow(row)
                    progress.update()
        except Exception as e:
            print(e)
        finally:
            [f.close() for f in fptrs]


        
