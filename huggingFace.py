from transformers import pipeline

# Load the sentiment analysis model
# summarizer = pipeline("summarization")

translator = pipeline('translation',model="facebook/nllb-200-distilled-600M")

# Perform sentiment analysis on a given text
text = '''
Korean battery makers announced a shared goal to expand the energy storage system (ESS) battery market. The companies are planning to compete in earnest with Chinese companies dominating the ESS market by waging a price war.
LG Energy Solution has succeeded in developing lithium ferrophosphate (LFP) battery cells for ESSs and will take the wraps off them for the first time at InterBattery 2023, the largest battery exhibition in Korea, to kick off this week, according to industry sources on March 12.
LG Energy Solution has been producing pouch-type nickel cobalt manganese (NCM) batteries for ESSs. Although ternary batteries boast high performance, LG Energy Solution will employ a strategy to expand the ESS market with LFP batteries as their unit price is high.
Chinese companies are leading the ESS market. According to SNE Research, Chinese battery companies accounted for 78.0 percent of the global ESS battery market in 2022. CATL, the No. 1 company in the world battery market, produced 530 GWh of ESS batteries last year, an increase of 212 percent compared to the previous year. Its market share reached 43.4 percent. The market shares of second-ranked BYD and third-ranked EVE posted shares of 11.5 percent and 7.8 percent, respectively.
In 2022, Korean companies logged a share of only 14.8 percent in the world ESS battery market. Last year, LG Energy Solution produced 9.2 GWh of batteries (a share of 7.5 percent and the 4th place), and Samsung SDI 8.9 GWh (7.3 percent and the 5th place). SK On does not produce batteries for ESSs.
When the commercialization of LFP batteries for ESSs is completed, LG Energy Solution will start full-scale production at its Nanjing plant in China and Ochang plant in Korea to compete with Chinese companies in the low-cost battery market.
Samsung SDI is also preparing to produce low-priced ESS batteries. The company is developing an NMX cathode material without expensive cobalt, and plans to mass-produce it for electric vehicle and ESS batteries when the development is completed. Currently, Samsung SDI is producing prismatic and cylindrical high-nickel nickel cobalt aluminum (NCA) batteries for ESSs.
Samsung SDI plans to increase its competitiveness in the ESS market with high growth potential by improving the performance of existing high-nickel products while preparing to produce low-priced products.
SK On does not produce batteries for ESSs. However, as the battery maker prepares to produce LFP batteries for electric vehicles, SK On will enter the ESS battery market in the future, industry insiders believe.
Cleaned text: Affordable Energy Storage Korean battery makers announced a shared goal to expand the energy storage system (ESS) battery market. The companies are planning to compete in earnest with Chinese companies dominating the ESS market by waging a price war. LG Energy Solution has succeeded in developing lithium ferrophosphate (LFP) battery cells for ESSs and will take the wraps off them for the first time at InterBattery 2023, the largest battery exhibition in Korea, to kick off this week, according to industry sources on March 12. LG Energy Solution has been producing pouch-type nickel cobalt manganese (NCM) batteries for ESSs. Although ternary batteries boast high performance, LG Energy Solution will employ a strategy to expand the ESS market with LFP batteries as their unit price is high. Chinese companies are leading the ESS market. According to SNE Research, Chinese battery companies accounted for 78.0 percent of the global ESS battery market in 2022. CATL, the No. 1 company in the world battery market, produced 530 
GWh of ESS batteries last year, an increase of 212 percent compared to the previous year. Its market share reached 43.4 percent. The market shares of second-ranked BYD and third-ranked EVE posted shares of 11.5 percent and 7.8 percent, respectively. In 2022, Ko to mass-produce it for electric vehicle and ESS batteries when the development is completed. Currently, Samsung SDI is producing prismatic and cylindrical high-nickel nickel cobalt aluminum (NCA) batteries for ESSs. Samsung SDd the 5th place). SK On does not pI plans to increase its competitiveness in the ESS market with high growth potential by improving the performance of existing high-nickel products while preparing to produce low-priced products. SK On does not produce batteriesompanies in the low-cost battery m for ESSs. However, as the battery maker prepares to produce LFP batteries for electric vehicles, SK On will enter the ESS battery market in the future, industry insiders believe.development is completed.
'''

kor_text = '올해 국내 자동차 회사들의 실적 전망치가 예상을 크게 웃돌 것으로 전망되었다.'

# Perform summarization on the text
# result = summarizer(text)
# print(result[0]['summary_text'])

# Perform translation on the text
result = translator(kor_text, src_lang="ko", tgt_lang="en", max_length=1000)
print(result[0]['translation_text'])
# %%
