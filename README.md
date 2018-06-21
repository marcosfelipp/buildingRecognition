# buildingRecognition
Algorithm to identify buildings in an image using Neural Networks

<h3>Download deeplab e tensorflow:</h3>
Instale o tensorflow e depois faça o clone do reposiório deeplab:

https://github.com/tensorflow/models/tree/master/research/deeplab

<h3> Download dataset: </h3>

Dataset usado para o treinamento é o Inria, que pode ser obtido no link:

https://project.inria.fr/aerialimagelabeling/download/

<h3>Pré-processsamento das imagens</h3>
As imagens originais do dataset estão em um tamanho muito grande. Não temos memória o suficiente para processar a imagem de uma vez
em nossa rede neural por isso temos que fazer recortes nas imagens.

Execute o codigo:

<h3>Convertendo o Dataset para formato do Tensorflow:</h3>

```
sudo python ./build_ade20k_data.py  \
  --train_image_folder="/home/ubuntu/images_austin/train/images" \
  --train_image_label_folder="/home/ubuntu/images_austin/train/label" \
  --val_image_folder="/home/ubuntu/images_austin/vallidation/images" \
  --val_image_label_folder="/home/ubuntu/images_austin/validation/label" \
  --output_dir="/home/ubuntu/images_austin/inria_dataset/tfrecord"
```
<h3>Algumas modificações no código do deeplab:</h3>

Acrescente o seguinte trecho no arquivo deeplab/datasets/segmentation_dataset.py Altere os campos train e val de acordo com a divisão do dataset

```
 _INRIA_INFORMATION = DatasetDescriptor(
    splits_to_sizes = {
        'train': 145, # num of samples in images/training
        'val': 35, # num of samples in images/validation
    },
    num_classes=1,
    ignore_label=255,
)
```
No mesmo código procure por _DATASETS_INFORMATION e acrescente a seguinte linha:
```
'inria': _INRIA_INFORMATION,
```
<h3>Fase de treinamento:</h3>

```
nohup python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=60000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size=625 \
    --train_crop_size=625 \
    --train_batch_size=32 \
    --min_resize_value=625 \
    --max_resize_value=625 \
    --fine_tune_batch_norm=True \
    --dataset="inria" \
    --initialize_last_layer=False \
    --last_layers_contain_logits_only=True \
    --tf_initial_checkpoint="/home/ubuntu/images_austin/results" \
    --train_logdir="/home/ubuntu/images_austin/results"\
    --dataset_dir="/home/ubuntu/images_austin/inria_dataset/tfrecord" &
    
```
<h3>Fase de avaliação:</h3>

```
nohup python deeplab/eval.py \
      --checkpoint_dir="/home/contatovidadeti/results4" \
      --eval_logdir="/home/contatovidadeti/results2/eval" \ 
      --dataset_dir="/home/contatovidadeti/inria_dataset/tfrecord" \
      --dataset="inria" --model_variant="xception_65" &
      
```
