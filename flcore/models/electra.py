
from transformers import ElectraForSequenceClassification, AutoTokenizer, ElectraConfig
from flcore.models.model_utils import BaseModel

class ELECTRA(BaseModel):
    def __init__(self, num_classes, num_embeddings, embedding_size, hidden_size, use_pt_model, is_seq2seq, dropout=0.1):
        super(ELECTRA, self).__init__()
        self.is_seq2seq = is_seq2seq
        self.tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")  # 使用ELECTRA的预训练模型
        
        if use_pt_model:  # fine-tuning
            config = ElectraConfig.from_pretrained('google/electra-small-discriminator', num_labels=num_classes)
            self.model = ElectraForSequenceClassification(config)

        else:  # from scratch
            config = ElectraConfig(
                num_labels=num_classes,
                hidden_size=hidden_size,
                vocab_size=num_embeddings,
                hidden_dropout_prob=dropout,
                # ELECTRA 模型配置通常不需要 `embedding_size` 参数
            )
            self.model = ElectraForSequenceClassification(config)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)