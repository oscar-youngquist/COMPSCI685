from transformers import pipeline
from tqdm import tqdm
import csv
from pathlib import Path

en_fr = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr", device=0)
fr_en = pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en", device=0)

paraphrase = lambda doc: fr_en(en_fr(doc)[0]['translation_text'])[0]["translation_text"]

if __name__ == "__main__":
    path = Path(__file__).absolute().parent.parent.parent / "SubSumE_Data"
    import pdb; pdb.set_trace()


    with open(path / 'processed_state_sentences.csv', 'r') as infile:
        with open(path / 'paraphrased_state_sentences.csv', 'w') as outfile:
            reader = csv.DictReader(infile)
            writer = csv.DictWriter(outfile, reader.fieldnames)
            writer.writeheader()
            i = 0
            for row in tqdm(reader):
                try:
                    row['sentence'] = paraphrase(row['sentence'])
                    writer.writerow(row)
                except Exception as e:
                    print(f"Failed to write sentence {row['sid']}")
                    row['sentence'] = '<ERR>'
                    writer.writerow(row)


        