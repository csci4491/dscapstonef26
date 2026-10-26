import numpy as np
import pandas as pd
import plotly.express as px

con_df = pd.read_csv('qwen_single_con.csv')
hyp_df = pd.read_csv('qwen_single_hyp.csv')
diffs_df = pd.read_csv('qwen_vector_diffs.csv')

def get_magnitude(df):
  df["magnitude"] = np.linalg.norm(df.loc[:,'dim_0':'dim_5119'], axis=1)
  return df

def main():
  global con_df, hyp_df, diffs_df
  diffs_df = get_magnitude(diffs_df)
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
      "Layer": diffs_df['Layer']
  })

  fig = px.scatter(data_frame=plot_df,
                   x="Hypothesis Magnitude",
                   y="Conclusion Magnitude",
                   color="Paper ID",
                   labels={"Hypothesis Magnitude": "Hypothesis Magnitude", "Conclusion Magnitude": "Conclusion Magnitude"},
                   hover_data={
                       "Difference Magnitude": True,
                       "Paper ID": True,
                       "Layer": True
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

if __name__ == "__main__":
  main()
