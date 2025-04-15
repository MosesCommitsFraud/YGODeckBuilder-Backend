# db/models.py
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship

Base = declarative_base()

class Card(Base):
    __tablename__ = 'cards'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String)
    human_readable_card_type = Column(String)
    frame_type = Column(String)
    desc = Column(Text)
    race = Column(String)
    archetype = Column(String)
    ygoprodeck_url = Column(String)
    is_staple = Column(Boolean, default=False)

    # Neu hinzugefügte Spalten:
    atk = Column(Integer)               # Angriff
    defense = Column(Integer)           # Verteidigung (Python-Schlüsselwort "def" vermeiden!)
    level = Column(Integer)             # Sterne / Level / Rang
    attribute = Column(String)          # Attribut (z.B. DARK, LIGHT, FIRE etc.)

    # Banstatus-Spalten
    ban_tcg = Column(String)   # z.B. "Limited", "Semi-Limited", "Banned", oder None
    ban_ocg = Column(String)
    ban_goat = Column(String)

    # Neue Spalten für referenzierte Effekte (als JSON-Text)
    referenced_cards = Column(Text)         # z.B. '["44818", "12345"]'
    referenced_archetypes = Column(Text)    # z.B. '["Starry Knight"]'
    referenced_races = Column(Text)         # z.B. '["Dragon"]'

    # Neue One-Hot Encoded Effekt-Kategorien (Boolean-Spalten)
    effect_search = Column(Boolean, default=False)
    effect_destroy = Column(Boolean, default=False)
    effect_negate = Column(Boolean, default=False)
    effect_draw = Column(Boolean, default=False)
    effect_special_summon = Column(Boolean, default=False)
    effect_banish = Column(Boolean, default=False)
    effect_send_gy = Column(Boolean, default=False)
    effect_recover_lp = Column(Boolean, default=False)
    effect_inflict_damage = Column(Boolean, default=False)
    effect_equip = Column(Boolean, default=False)
    effect_modify_stats = Column(Boolean, default=False)
    effect_protect = Column(Boolean, default=False)
    effect_discard = Column(Boolean, default=False)
    effect_change_position = Column(Boolean, default=False)
    effect_return = Column(Boolean, default=False)
    effect_shuffle = Column(Boolean, default=False)
    effect_copy = Column(Boolean, default=False)
    effect_counter = Column(Boolean, default=False)
    effect_token_summon = Column(Boolean, default=False)
    effect_deck_manipulation = Column(Boolean, default=False)

    sets = relationship("CardSet", back_populates="card", cascade="all, delete-orphan")
    images = relationship("CardImage", back_populates="card", cascade="all, delete-orphan")
    prices = relationship("CardPrice", back_populates="card", cascade="all, delete-orphan")
    deck_cards = relationship("DeckCard", back_populates="card", cascade="all, delete-orphan")


class CardSet(Base):
    __tablename__ = 'card_sets'
    id = Column(Integer, primary_key=True)
    card_id = Column(Integer, ForeignKey('cards.id'))
    set_name = Column(String)
    set_code = Column(String)
    set_rarity = Column(String)
    set_rarity_code = Column(String)
    set_price = Column(String)

    card = relationship("Card", back_populates="sets")


class CardImage(Base):
    __tablename__ = 'card_images'
    id = Column(Integer, primary_key=True)
    card_id = Column(Integer, ForeignKey('cards.id'))
    image_url = Column(String)
    image_url_small = Column(String)
    image_url_cropped = Column(String)

    card = relationship("Card", back_populates="images")


class CardPrice(Base):
    __tablename__ = 'card_prices'
    id = Column(Integer, primary_key=True)
    card_id = Column(Integer, ForeignKey('cards.id'))
    cardmarket_price = Column(String)
    tcgplayer_price = Column(String)
    ebay_price = Column(String)
    amazon_price = Column(String)
    coolstuffinc_price = Column(String)

    card = relationship("Card", back_populates="prices")


class Deck(Base):
    __tablename__ = 'decks'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    deck_cards = relationship("DeckCard", back_populates="deck", cascade="all, delete-orphan")


class DeckCard(Base):
    __tablename__ = 'deck_cards'
    id = Column(Integer, primary_key=True)
    deck_id = Column(Integer, ForeignKey('decks.id'), nullable=False)
    card_id = Column(Integer, ForeignKey('cards.id'), nullable=False)
    section = Column(String, nullable=False)  # "main", "extra" oder "side"

    deck = relationship("Deck", back_populates="deck_cards")
    card = relationship("Card", back_populates="deck_cards")

