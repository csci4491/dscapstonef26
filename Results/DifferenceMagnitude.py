import numpy as np
import pandas as pd
import plotly.express as px

con_df = pd.read_csv('qwen_single_con.csv')
hyp_df = pd.read_csv('qwen_single_hyp.csv')
diffs_df = pd.read_csv('qwen_vector_diffs.csv')

def get_magnitude(df):
  df["magnitude"] = np.linalg.norm(df.loc[:,'dim_0':'dim_5119'], axis=1)
  return df

def detect_outliers(df, threshold=3):
  """
  Detect magnitude outliers using the Median Absolute Deviation (MAD) method.
  """
  df= df.copy()

  df["magnitude_robust_z"]=0.0
  df["magnitude_outlier"]="Normal"

  # Analyze each layer separately
  for layer, group in df.groupby("Layer"):
      magnitudes = group["magnitude"]

      median= magnitudes.median()

      mad= np.median(np.abs(magnitudes - median))

      if mad==0:
        continue # Avoid division by zero

      robust_z= ( 0.6745 * (magnitudes - median) / mad )

      print(
        f"  Max |robust z| = "
        f"{robust_z.abs().max():.4f}"
        )


      df.loc[group.index, "magnitude_robust_z"] = robust_z

      df.loc[group.index[robust_z>threshold], "magnitude_outlier"] = "High"

      df.loc[group.index[robust_z<-threshold], "magnitude_outlier"] = "Low"

      return df

def main():
  global con_df, hyp_df, diffs_df
  diffs_df = get_magnitude(diffs_df)

  # Magnitude Outlier Detection
  diffs_df= detect_outliers(diffs_df)
  outlier_df= diffs_df[diffs_df["magnitude_outlier"] != "Normal"].copy()
  
  print(f'Difference Magnitude for Each Vector: {diffs_df.shape} \n{diffs_df["magnitude"]}')

  # get magnitude for hypothesis and conclusion for graphing purposes
  hyp_df["magnitude"] = np.linalg.norm(hyp_df.loc[:,'dim_0':'dim_5119'], axis=1)
  con_df["magnitude"] = np.linalg.norm(con_df.loc[:,'dim_0':'dim_5119'], axis=1)

  # plot magnitudes and display difference within the labels
  plot_df = pd.DataFrame({
      "Hypothesis Magnitude": hyp_df["magnitude"],
      "Conclusion Magnitude": con_df["magnitude"],
      "Paper ID": hyp_df['Paper_ID'],
      "Difference Magnitude": diffs_df["magnitude"],
      "Layer": diffs_df['Layer'],
      "Magnitude Outlier":diffs_df["magnitude_outlier"],
      'Robust Z Score': diffs_df["magnitude_robust_z"]
  })

  fig = px.scatter(data_frame=plot_df,
                   x="Hypothesis Magnitude",
                   y="Conclusion Magnitude",
                   color="Paper ID",
                   labels={"Hypothesis Magnitude": "Hypothesis Magnitude", "Conclusion Magnitude": "Conclusion Magnitude"},
                   hover_data={
                       "Difference Magnitude": True,
                       "Paper ID": True,
                       "Layer": True,
                       "Robust Z Score": True
                   },
                   title="Hypothesis and Conclusion Magnitude")

  fig.show()
  fig.write_html("hypothesis_conclusion_magnitude.html")

  
  # Heatmap for specific layers using a pivot
  pivot_df = plot_df.pivot(index='Layer', columns='Paper ID', values='Difference Magnitude')
  fig = px.imshow(pivot_df,
                  labels=dict(x="Paper ID", y="Layer", color="Difference Magnitude"),
                  x=pivot_df.columns,
                  y=pivot_df.index,
                  text_auto=True,
                  color_continuous_scale='Viridis',
                  aspect = "auto")
  fig.update_layout(title_text='Difference Magnitude by Layer and Paper ID',
                    coloraxis_showscale=False)
  fig.show()
  fig.write_html("heatmap_difference_magnitude.html")

  # save all magnitudes
  diffs_df.to_csv("difference_magnitudes_with_outliers.csv", index=False)
  # save only outliers
  outlier_df.to_csv("magnitude_outliers.csv", index=False)

  print("\n Magnitude Outliers:")
  print(
    outlier_df[["Paper_ID", "Layer", "magnitude", "magnitude_robust_z", "magnitude_outlier"]]
  )

if __name__ == "__main__":
  main()
