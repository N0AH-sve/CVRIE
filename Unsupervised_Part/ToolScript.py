def extract_testimony_text(raw_line: str) -> str | None:
    raw_line = raw_line.rstrip("\n")

    if not raw_line:
        return None

    try:
        _record_id, rest_of_line = raw_line.split(",", 1)
    except ValueError:
        return None

    rest_of_line = rest_of_line.lstrip()
    if rest_of_line.startswith("0x000000,"):
        rest_of_line = rest_of_line[len("0x000000,"):].lstrip()

    rest_of_line = rest_of_line.strip()

    if len(rest_of_line) >= 2 and rest_of_line[0] == '"' and rest_of_line[-1] == '"':
        rest_of_line = rest_of_line[1:-1].replace('""', '"')

    return rest_of_line or None

def parse_testimonies(input_csv_path, initial_record_count):
    extracted_testimonies = []
    with open(input_csv_path, "r", encoding="utf-8") as file:
        for line in file:
            testimony_text = extract_testimony_text(line)
            if testimony_text:
                extracted_testimonies.append(testimony_text)

    records_lost_during_parsing = initial_record_count - len(extracted_testimonies)

    print(f"\n✓ Parsing complete")
    print(f"  Testimonies extracted: {len(extracted_testimonies)}")
    print(f"  Records lost (invalid format): {records_lost_during_parsing}")

    return extracted_testimonies