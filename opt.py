import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--batch_size", type=int, default=64,
                        help='Data batch size')
    parser.add_argument("--block_size", type=int, default=256,
                        help='Width of attention')                        
    parser.add_argument("--number_head", type=int, default=6,
                        help='Number of attention head')                            
    parser.add_argument("--number_block", type=int, default=6,
                        help='Number of decode block')  
    parser.add_argument("--embedded_size", type=int, default=32*6,
                        help='Embedding size')                                                  
    parser.add_argument("--eval_interval", type=int, default=500,
                        help='Evaluate mode every number of step')                           
    parser.add_argument("--learning_rate", type=float, default=4e-3,
                        help='Learning rate')
    parser.add_argument("--generation", type=int, default=0,
                        help='Generate text')               
    parser.add_argument("--train_only", type=int, default=0,
                        help='Train model')                                       
    parser.add_argument("--training_step", type=int, default=500,
                        help='Number of step for training')                           

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()                                                