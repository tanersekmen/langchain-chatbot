from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv


information = """
Mustafa Kemal Atatürk[c] (1881,[d] Selanik, Osmanlı İmparatorluğu - 10 Kasım 1938, İstanbul, Türkiye), Türk mareşal, devlet adamı, yazar, Türk Kurtuluş Savaşı'nın başkomutanı, Türkiye Cumhuriyeti'nin kurucusu ve ilk cumhurbaşkanıdır.

I. Dünya Savaşı sırasında Osmanlı ordusunda görev yapan Atatürk, Çanakkale Cephesi'nde miralaylığa, Sina ve Filistin Cephesi'nde ise Yıldırım Ordular Grubu komutanlığına atandı. Savaşın sonunda, Osmanlı İmparatorluğu'nun yenilgisini izleyen Kurtuluş Savaşı ile simgelenen Anadolu Hareketi'ne öncülük ve önderlik etti. Türk Kurtuluş Savaşı sürecinde Ankara Hükûmeti'ni kurdu, Türk Orduları Başkomutanı olarak Sakarya Meydan Muharebesi'ndeki başarısından dolayı 19 Eylül 1921 tarihinde "gazi" sanını aldı ve mareşallik rütbesine yükseldi. Askerî ve siyasal eylemleriyle İtilaf Devletleri ve destekçilerine karşı yengi kazandı. Savaşın ardından Cumhuriyet Halk Partisini "Halk Fırkası" adıyla kurdu ve ilk genel başkanı oldu. 29 Ekim 1923'te Cumhuriyetin İlanı ardından cumhurbaşkanı seçildi. 1938'deki ölümüne dek dört dönem bu görevi yürütmüş olup günümüze değin Türkiye'de en uzun süre cumhurbaşkanlığı yapmış kişidir.

Atatürk; çağdaş, ilerici ve laik bir ulus devlet kurmak için siyasal, ekonomik ve kültürel alanlarda sekülarist ve milliyetçi nitelikte yenilikler gerçekleştirdi. Yabancılara tanınan ekonomik ayrıcalıklar kaldırıldı ve onlara ait üretim araçları ve demir yolları millîleştirildi. Tevhîd-i Tedrîsât Kanunu ile eğitim, Türk hükûmetinin denetimine girdi. Seküler ve bilimsel eğitim esas alındı. Binlerce yeni okul yapıldı. İlköğretim ücretsiz ve zorunlu duruma getirildi. Yabancı okullar devlet denetimine alındı. Köylülerin sırtına yüklenen ağır vergiler azaltıldı. Erkeklerin serpuşlarında ve giysilerinde bazı değişiklikler yapıldı. Takvim, saat ve ölçülerde değişikliklere gidildi. Mecelle kaldırılarak yerine seküler Türk Kanunu Medenisi yürürlüğe konuldu. Kadınların sivil ve siyasal hakları pek çok Batı ülkesinden önce tanındı. Çok eşlilik yasaklandı. Kadınların tanıklığı ve miras hakkı, erkeklerinkiyle eşit duruma getirildi. Benzer olarak, dünyanın çoğu ülkesinden önce olarak Türkiye'de kadınlara ilkin yerel seçimlerde (1930), sonra genel seçimlerde (1934) seçme ve seçilme hakkı tanındı. Ceza ve borçlar hukukunda seküler yasalar yürürlüğe konuldu. Sanayi Teşvik Kanunu kabul edildi. Toprak reformu için çabalandı. Arap harfleri temelli Osmanlı alfabesinin yerine Latin harfleri temelli yeni Türk alfabesi kabul edildi. Halkı okuryazar kılmak için eğitim seferberliği başlatıldı. Üniversite Reformu gerçekleştirildi. Birinci Beş Yıllık Sanayi Planı yürürlüğe konuldu. Sınıf ve durum ayrımı gözeten lakap ve unvanlar kaldırıldı ve soyadları yürürlüğe konuldu. Bağdaşık ve birleşmiş bir ulus yaratılması için Türkleştirme siyaseti yürütüldü.
"""

if __name__ == '__main__':
    load_dotenv()
    summary_template = f"""
    sana verilen {information} ile birlikte şu sorulara cevap vermeni istiyorum.
    1. verilen text için özet
    2. bu bilgiler içerisinde en önemli özellik
    """

    summary_prompt_template = PromptTemplate(
        input_variables=['information'], template = summary_template
    )

    llm = ChatOpenAI(temperature = 0, model_name = "gpt-3.5-turbo")
    chain = LLMChain(llm = llm, prompt = summary_prompt_template)
    print(chain.run(information = information))
