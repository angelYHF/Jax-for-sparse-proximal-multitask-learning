from checkpoint import Checkpoint
from tools import protect


class ModelTrainedOrOnTraining(Exception):
    pass


class BaseModel(metaclass=protect("train", "__init__")):
    """
    Only Take care of 'started' and 'finished' state lock, as well as identityFields(params)
    """

    # Public interface method
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    # Public interface method
    def train(self, *args, **kwargs):
        self.__lazy_initialize__()
        try:
            self.__init_training__()
            self._train_(*args, **kwargs)
            self.__close_training__()
        except ModelTrainedOrOnTraining:
            return

    # Parent private method
    def __lazy_initialize__(self):
        params = self.kwargs['params']
        repo = self.kwargs['configs']['repo']
        overrideUnfinished = self.kwargs['configs']['overrideUnfinished']
        self.trainedOrOnTraining = False
        try:
            model = repo.pick(**params)
            if model['finished']:
                self.trainedOrOnTraining = True
                return
            elif model['started'] and not overrideUnfinished:
                self.trainedOrOnTraining = True
                return
            else:
                pass
        except FileNotFoundError:
            pass

        self.checkpoint = Checkpoint(repo.pick(**params, pathOnly=True), override=overrideUnfinished)
        self.checkpoint.register_fields([(k, v) for (k, v) in params.items()])
        self.checkpoint.register_fields([
            ("started", False),
            ("finished", False),
        ])
        self._initialize_(*self.args, **self.kwargs)

    # Parent private method
    def __init_training__(self):
        if self.trainedOrOnTraining:
            raise ModelTrainedOrOnTraining

        self.checkpoint["started"] = True
        self.checkpoint.save()

    # Parent private method
    def __close_training__(self):
        self.checkpoint["finished"] = True
        self.checkpoint.save()

    # Protect method
    def _initialize_(self, *args, **kwargs):
        raise NotImplementedError

    # Protect method
    def _train_(self, *args, **kwargs):
        raise NotImplementedError
