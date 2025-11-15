import torch
from transformers import MobileBertForSequenceClassification, AutoTokenizer, MobileBertConfig
from flcore.models.model_utils import BaseModel



class MobileBert(BaseModel):
    def __init__(self, num_classes, num_embeddings, embedding_size, hidden_size, use_pt_model, is_seq2seq, dropout=0.1):
        super(self).__init__()
        self.is_seq2seq = is_seq2seq
        self.tokenizer = AutoTokenizer.from_pretrained("lordtt13/emo-mobilebert")
        
        if use_pt_model: # fine-tuning
            config = MobileBertConfig.from_pretrained('lordtt13/emo-mobilebert', num_labels=num_classes)
            self.model = MobileBertForSequenceClassification(config)

        else: # from scratch
            config = MobileBertConfig(
                num_labels=num_classes,
                hidden_size=hidden_size,
                vocab_size=num_embeddings,
                hidden_dropout_prob=dropout,
                intermediate_size=embedding_size,
            )
            self.model = MobileBertForSequenceClassification(config)

    def forward(self, input_ids, attention_mask):

        return self.model(input_ids, attention_mask)