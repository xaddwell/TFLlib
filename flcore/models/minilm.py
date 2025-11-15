
from transformers import LongformerForSequenceClassification, AutoTokenizer, LongformerConfig
from flcore.models.model_utils import BaseModel

class MiniLM(BaseModel):
    def __init__(self, num_classes, num_embeddings, embedding_size, hidden_size, use_pt_model, is_seq2seq, dropout=0.1):
        super(self).__init__()
        self.is_seq2seq = is_seq2seq
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096") 
        # 使用MiniLM的预训练模型
        
        if use_pt_model:  # fine-tuning
            config = LongformerConfig.from_pretrained('allenai/longformer-base-4096', num_labels=num_classes)
            self.model = LongformerForSequenceClassification(config)

        else:  # from scratch
            config = LongformerConfig(
                num_labels=num_classes,
                hidden_size=hidden_size,
                vocab_size=num_embeddings,
                hidden_dropout_prob=dropout,
                # Longformer 的配置可能不需要 embedding_size 参数
            )
            self.model = LongformerForSequenceClassification(config)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)