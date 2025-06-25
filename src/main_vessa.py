"""Main file for launching Vessa trainings."""

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
import vessa_dataset  # pylint: disable=unused-import
import ops  # pylint: disable=unused-import
#import trainer_vessa as trainer
import trainer_vessa as trainer
from scenic.train_lib import train_utils
import wandb

FLAGS = flags.FLAGS


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main entry point for vessa training."""
  data_rng, rng = jax.random.split(rng)
  
  print(config)
  
  wandb.init(
      # set the wandb project where this run will be logged
      project=config.project,
      name=config.experiment_name,
      # track hyperparameters and run metadata with wandb.config
      config=config.to_dict()
  )
  
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  
  #dataset_val = train_utils.get_dataset(
  #    config.val, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  # Start a run, tracking hyperparameters
  

  trainer.train(
      rng=rng,
      config=config,
      dataset=dataset,
      #dataset_val=dataset_val,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main=main)

#fazer o google clound terminal nao fechar com a execução -> tmux na configuração
#Rodar o train
#Corrigir arquivo de entrada do 