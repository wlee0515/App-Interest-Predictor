import pandas as pd
import numpy as np

class InterestVectorConvertor():
    def __init__(self, df_Category):       
        self.df_Category = df_Category.copy()
        self.list_exclude = ["label_id", "category", "category-mod"]
        self.list_VectorColumn = [ x for x in self.df_Category.columns if x not in self.list_exclude]

    def convertVector(self, interestVectorPrediction):
        wWorkingMatrix = self.df_Category.drop(self.list_exclude, axis=1).to_numpy()
        wScore = np.matmul(wWorkingMatrix, np.transpose(interestVectorPrediction))
        wdf = self.df_Category["category-mod"].to_frame();
        wdf_score = pd.DataFrame(wScore, columns =["category-score"])
        wdf = pd.concat([wdf, wdf_score], axis=1)
        wdf = wdf.groupby("category-mod").mean()
        wdf = wdf.sort_values("category-score", ascending=False).head(25)
        return wdf
        