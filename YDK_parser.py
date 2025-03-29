def parse_ydk_file(file_path):
    """
    Liest eine .ydk-Datei und gibt dict mit den Keys 'main', 'extra', 'side' zurück,
    jeweils Listen von Card-IDs als Strings.

    Format-Beispiel:
    #created by ...
    #main
    89631139
    89631139
    ...
    #extra
    ...
    !side
    ...
    """
    main_cards = []
    extra_cards = []
    side_cards = []

    current_section = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "main" in line.lower():
                    current_section = "main"
                elif "extra" in line.lower():
                    current_section = "extra"
                else:
                    current_section = None
            elif line.startswith("!"):
                if "side" in line.lower():
                    current_section = "side"
                else:
                    current_section = None
            else:
                # Falls es eine ID ist
                if current_section == "main":
                    main_cards.append(line)
                elif current_section == "extra":
                    extra_cards.append(line)
                elif current_section == "side":
                    side_cards.append(line)

    return {
        "main": main_cards,
        "extra": extra_cards,
        "side": side_cards
    }
