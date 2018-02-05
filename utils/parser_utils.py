class ParserClass(object):
    def __init__(self, parser):
        """
        Parses arguments and saves them in the Parser Class
        :param parser: A parser to get input from
        """
        parser.add_argument('--batch_size', nargs="?", type=int, default=64, help='batch_size for experiment')
        parser.add_argument('--epochs', type=int, nargs="?", default=100, help='Number of epochs to train for')
        parser.add_argument('--logs_path', type=str, nargs="?", default="classification_logs/",
                            help='Experiment log path, '
                                 'where tensorboard is saved, '
                                 'along with .csv of results')
        parser.add_argument('--experiment_prefix', nargs="?", type=str, default="classification",
                            help='Experiment name without hp details')
        parser.add_argument('--continue_epoch', nargs="?", type=int, default=-1, help="ID of epoch to continue from, "
                                                                                      "-1 means from scratch")
        parser.add_argument('--tensorboard_use', nargs="?", type=str, default="False",
                            help='Whether to use tensorboard')
        parser.add_argument('--dropout_rate', nargs="?", type=float, default=0.35, help="Dropout value")
        parser.add_argument('--batch_norm_use', nargs="?", type=str, default="False", help='Whether to use tensorboard')
        parser.add_argument('--strided_dim_reduction', nargs="?", type=str, default="False",
                            help='Whether to use tensorboard')
        parser.add_argument('--seed', nargs="?", type=int, default=1122017, help='Whether to use tensorboard')

        self.args = parser.parse_args()

    def get_argument_variables(self):
        """
        Processes the parsed arguments and produces variables of specific types needed for the experiments
        :return: Arguments needed for experiments
        """
        batch_size = self.args.batch_size
        experiment_prefix = self.args.experiment_prefix
        strided_dim_reduction = True if self.args.strided_dim_reduction == "True" else False
        batch_norm = True if self.args.batch_norm_use == "True" else False
        seed = self.args.seed
        dropout_rate = self.args.dropout_rate
        tensorboard_enable = True if self.args.tensorboard_use == "True" else False
        continue_from_epoch = self.args.continue_epoch  # use -1 to start from scratch
        epochs = self.args.epochs
        logs_path = self.args.logs_path

        return batch_size, seed, epochs, logs_path, continue_from_epoch, tensorboard_enable, batch_norm, \
               strided_dim_reduction, experiment_prefix, dropout_rate

class KerasParserClass(object):
    def __init__(self, parser):
        """
        Parses arguments and saves them in the Parser Class
        :param parser: A parser to get input from
        """
        parser.add_argument('--batch_size', nargs="?", type=int, default=64, help='batch_size for experiment')
        parser.add_argument('--epochs', type=int, nargs="?", default=100, help='Number of epochs to train for')
        parser.add_argument('--logs_path', type=str, nargs="?", default="classification_logs/",
                            help='Experiment log path')
        parser.add_argument('--experiment_prefix', nargs="?", type=str, default="classification",
                            help='Experiment name without hp details')
        parser.add_argument('--continue_epoch', nargs="?", type=int, default=-1, help="ID of epoch to continue from, "
                                                                                      "-1 means from scratch")
        parser.add_argument('--dropout_rate', nargs="?", type=float, default=0.35, help="Dropout value")
        parser.add_argument('--batch_norm_use', nargs="?", type=str, default="False", help='Whether to use batch norm')
        parser.add_argument('--n_layers', nargs="?", type=int, default="1", help='Number hidden layers')
        parser.add_argument('--max_len', nargs="?", type=int, default="1", help='Max number of words')
        parser.add_argument('--seed', nargs="?", type=int, default=1122017, help='Random seed to use')

        self.args = parser.parse_args()

    def get_argument_variables(self):
        """
        Processes the parsed arguments and produces variables of specific types needed for the experiments
        :return: Arguments needed for experiments
        """
        batch_size = self.args.batch_size
        experiment_prefix = self.args.experiment_prefix
        batch_norm = True if self.args.batch_norm_use == "True" else False
        seed = self.args.seed
        dropout_rate = self.args.dropout_rate
        continue_from_epoch = self.args.continue_epoch  # use -1 to start from scratch
        epochs = self.args.epochs
        max_len = self.args.max_len
        n_layers = self.args.n_layers
        logs_path = self.args.logs_path

        return batch_size, seed, epochs, logs_path, continue_from_epoch, batch_norm, \
               experiment_prefix, dropout_rate, n_layers, max_len
