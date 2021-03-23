# https://docs.allennlp.org/v1.0.0rc3/tutorials/getting_started/using_pretrained_models/

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

sentence="ENGLISH FISHING BOAT CAPTURED.—The French torpedo-boat No. 89, while making a run round the port of Havre during Wednesday night, captured the English fishing boat Countess within the three mile limit and took her into port."

newstext="ENGLISH FISHING BOAT CAPTURED.—The French torpedo-boat No. 89, while making a run round the port of Havre during Wednesday night, captured the English fishing boat Countess within the three mile limit and took her into port.—Temps.  SMALLPOX iKCREASiKG.—The smallpox outbreak in London is evidently of a virulent type. The death rate for the present epidemic works out at 24 per cent., or almost one in every four cases. The number of patients at present under treatment is 183. Six fresh cases are reported. —Evening News.  ROMANO'S AT A¡:;OTION.-Romano's Restaurant was put up at auction by order of the Court of Chancery. Although there was a very large attendance at Tokenhouse Yard, only .one bid of £20,000 was made, and as this did not come up to the reserve fixed by the judge the property was withdrawn.—Pall Mall Gazette.  TRIPLETS GO Ml:GLEss.-When Governor Stanley, of Kansas, was elected, he made a solemn promise to give silver mugs to all triplets born in Kansas during his administration. After providing for thirty-seven sets of triplets, Governor Stanley has gone on strike, and refused to buy any more. He says that the triplet boom is greater than he expected, and unfair to him.—Evening News.  GIRL ARTIST'S BANKRUPTCY. — The affairs of Miss Ellen Mortlock were dealt with before the Official Receiver in the Bankruptcy Court. The debtor, a portrait painter, lately residing in Sloane street, Chelsea, said that she owed about £1,000, and that her assets consisted of two claims of £1,9°0 each upon the Shah of Persia and LI-Hung-Chang. To her inability to obtain payment of these debts the debtor ascribed her insolvency. The meeting was adjourned for a fortnight. —Pall Mall Gazette."

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/fine-grained-ner.2021-02-11.tar.gz")
results = predictor.predict(
    sentence=newstext
)
for word, tag in zip(results["words"], results["tags"]):
    print(f"{word}\t{tag}")
