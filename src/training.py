from typing import List, Dict
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from src.config import NUM_EPOCHS
from src.embedding import get_device

def create_training_examples(pre_df, category_descriptions: Dict[str, str]) -> List[InputExample]:
    examples = []
    for _, row in pre_df.iterrows():
        user_text = str(row["abstract"])
        instr = str(row["clean_instrument"])
        item_text = f"{instr}: {category_descriptions.get(instr, instr)}"
        examples.append(InputExample(texts=[user_text, item_text]))
    return examples

def train_model(model_name: str, train_examples: List[InputExample]) -> SentenceTransformer:
    device = get_device()
    print(f"[INFO] Training on device: {device}")
    model = SentenceTransformer(model_name)
    model.to(device)

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=16, drop_last=False)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # new .fit API (stable); keep epochs low on MPS
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=NUM_EPOCHS,
        show_progress_bar=True,
        warmup_steps=min(100, max(10, len(train_loader)//10)),
    )
    return model
