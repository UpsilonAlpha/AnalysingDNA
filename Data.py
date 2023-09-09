import pandas as pd
import plotly.express as px

#370 Tumor Samples
#50 Normal Tissue Samples
#420 Total Samples
rnaseq = pd.read_csv("G9_liver_dna-meth.csv", index_col="Label")
rows = rnaseq.loc["Solid Tissue Normal"]
print(rows)

geneMean = rows.mean()

print(geneMean)

fig = px.scatter(geneMean, log_y=True)

fig.show()

#Save 15 normal tissue for testing
#Save 111 tumor samples for testing

# mean = 0
# for column in rnaseq:
#     print(column)
#     for ind in rnaseq.index:
#         if rnaseq["Label"][ind] == "Solid Tissue Normal":
#             mean = mean + column[ind]
#     mean = (mean/50)
#     print(mean)

