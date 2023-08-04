FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime as pytorch
COPY ./input_data/influenza_human_PPN_clean.gml /input_data/
COPY ./input_data/feature_mtx.pkl /input_data/
COPY ./*.py /app/ho-vgae_ppi_predictor/
COPY requirements.txt /app/ho-vgae_ppi_predictor/

RUN pip install -r /app/ho-vgae_ppi_predictor/requirements.txt

ENTRYPOINT [ "python","/app/ho-vgae_ppi_predictor/train.py","--path_to_graph=/input_data/influenza_human_PPN_clean.gml","--path_to_node_features=/input_data/feature_mtx.pkl","--model=HOVGAE","--alpha=0.2" ]