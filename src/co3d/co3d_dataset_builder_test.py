"""Co3D dataset."""

from . import co3d_dataset_builder_partner
import tensorflow_datasets as tfds

class Co3DTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for Co3D dataset."""
  # TODO(Co3D):
  DATASET_CLASS = co3d_dataset_builder_partner.Builder
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
