

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from flcore.models.model_utils import BaseModel

class TinyBert(BaseModel):
    def __init__(self, num_classes, num_embeddings, embedding_size, hidden_size, use_pt_model=True, dropout=0.1):
        super(TinyBert, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2-v2")
        # 使用TinyBert的预训练模型
        config = AutoConfig.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-2-v2', num_labels=num_classes)
        self.model = AutoModelForSequenceClassification.from_config(config)

    def forward(self, input_ids, attention_mask):

        return self.model(input_ids, attention_mask=attention_mask)
    