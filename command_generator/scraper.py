import subprocess
import re
from utils import TYPE_MAPS, ARG_TYPES, MANUAL_SYNTAX_INSERTS
import json
from pprint import pprint
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os


class ManScraper:
    def __init__(self, utilities):

        self.utilities = utilities
        self.descs = {}
        self.data = {}
        self.relevant_flags = {}
        self.lock = threading.Lock()

    def load_relevant_flags(self, path='original_training.txt'):
        with open(path) as fp:
            self.relevant_flags = set(fp.read().replace('\n', ' ').split(' '))

    def is_relevant(self, flag):
        return False if "--" in flag and flag not in self.relevant_flags else True

    def display_structures(self):
        pprint(self.descs)
        pprint(self.data)

    def _get_man_page(self, utility):
        try:
            for section in [1, 2, 3, 4, 5, 6, 7, 8]:
                try:
                    result = subprocess.run(
                        ['man', str(section), utility],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        return result.stdout
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
                    
            result = subprocess.run(
                ['man', utility],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout
                
        except Exception as e:
            print(f"Error getting man page for {utility}: {str(e)}")
            
        return None

    def _process_utility(self, utility):
        man_page = self._get_man_page(utility)
        if not man_page:
            return utility, None, None

        try:
            syntax_match = re.search(r'SYNOPSIS.*?(?=DESCRIPTION|$)', man_page, re.DOTALL | re.IGNORECASE)
            syntax = syntax_match.group(0) if syntax_match else ""
            
            options_match = re.search(r'OPTIONS.*?(?=EXAMPLES|EXIT STATUS|FILES|SEE ALSO|$)', man_page, re.DOTALL | re.IGNORECASE)
            options = options_match.group(0) if options_match else ""
            
            flag_lines = re.findall(r'^\s*(-\w+.*?)$', options, re.MULTILINE)
            
            synopsis_flags = re.findall(r'(-\w+)', syntax)
            flag_lines.extend(synopsis_flags)
            
            return utility, syntax, set(flag_lines)
            
        except Exception as e:
            print(f"Error parsing man page for {utility}: {str(e)}")
            return utility, None, None

    def extract_utilities(self):
        print(f"Parsing local man pages for {len(self.utilities)} utilities...")

        no_page_uts, no_syntax_uts = [], []
        successful_searches = []

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = {executor.submit(self._process_utility, utility): utility for utility in self.utilities}
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                utility = futures[future]
                try:
                    result = future.result()
                    utility, syntax, flag_data = result
                    
                    if syntax is None:
                        if utility in MANUAL_SYNTAX_INSERTS:
                            with self.lock:
                                self.descs[utility] = MANUAL_SYNTAX_INSERTS[utility]
                            successful_searches.append(utility)
                        else:
                            no_syntax_uts.append(utility)
                    else:
                        # Генерируем синтаксис на основе man-страницы
                        cleaned_syntax = self._generate_syntax(utility, syntax)
                        if cleaned_syntax:
                            with self.lock:
                                self.descs[utility] = cleaned_syntax
                            successful_searches.append(utility)
                        else:
                            no_syntax_uts.append(utility)

                    if flag_data is not None:
                        self._clean_and_insert_flags(utility, flag_data)

                except Exception as e:
                    no_page_uts.append(utility)
                    print(f"Error processing {utility}: {str(e)}")

        self.convert_flag_types()
        if 'find' in self.data:
            self.data['find -L'] = self.data['find']

        print(f"Successfully parsed {len(successful_searches)} utilities")
        print(f"{len(no_page_uts)} utilities with no man page: {no_page_uts}")
        print(f"{len(no_syntax_uts)} utilities without syntax structures: {no_syntax_uts}")

    def _clean_and_insert_flags(self, utility, lines):
        with self.lock:
            if utility not in self.data:
                self.data[utility] = {}
                
            for flag_line in lines:
                flag_line = self._remove_punctuation(flag_line)
                flag, arg = self._get_flag(flag_line), None
                
                if "[" in flag_line and "]" in flag_line:
                    arg = self._get_inner_brackets(flag_line)
                elif "=" in flag_line:
                    arg = self._get_equal_arg(flag_line)
                elif len(flag_line.split(" ")) == 2 and "-" not in flag_line.split(" ")[1]:
                    arg = flag_line.split(" ")[1]
                
                if (flag not in self.data[utility] or not self.data[utility][flag]) and self.is_relevant(flag):
                    self.data[utility][flag] = arg

    @staticmethod
    def _generate_syntax(utility, syntax):
        if not syntax or 'option' not in syntax.lower():
            return None

        s = syntax.replace('...', '')
        s = ManScraper._remove_brackets(ManScraper._remove_punctuation(s)).lower()

        sp = s.split(" ")
        cleaned = []
        for val in sp:
            if val == utility:
                cleaned.append(val)
            elif "option" in val:
                cleaned.append("[Options]")
            elif val and val[0] != "-":
                s = ""
                for data_type in TYPE_MAPS:
                    for match in TYPE_MAPS[data_type]:
                        if match == val:
                            s = data_type
                if s:
                    if s == '[File]' and s in cleaned:
                        cleaned.append('[File2]')
                    else:
                        cleaned.append(s)

        return " ".join(cleaned)

    def save_json(self, syntax_path='syntax.json', map_path='utility_map.json'):
        if not self.data or not self.descs:
            raise Exception("Mapping and syntax uninitialized")

        with open(syntax_path, 'w') as fp:
            json.dump(self.descs, fp)

        with open(map_path, 'w') as fp:
            json.dump(self.data, fp)

    def insert_syntax(self, utility, syntax):
        self.descs[utility] = syntax

    def insert_flag(self, utility, flag, arg):
        self.data[utility][flag] = arg

    def unprocessed_utilities(self):
        ret = []
        for utility in self.utilities:
            if utility not in self.descs:
                ret.append(utility)
        return ret

    @staticmethod
    def _get_inner_brackets(s):
        open_idx = s.index("[") + 1
        closed_idx = s.index("]")
        a = s[open_idx: closed_idx]
        a = a.replace("=", "")
        return a

    @staticmethod
    def _get_equal_arg(s):
        return ManScraper._remove_punctuation(s.split("=")[1])

    @staticmethod
    def _remove_punctuation(s):
        punctuation = set(_ for _ in ",.()")
        return "".join([x if x not in punctuation else "" for x in s])

    @staticmethod
    def _remove_brackets(s):
        brackets = {"[", "]"}
        return "".join([x if x not in brackets else "" for x in s])

    @staticmethod
    def _get_flag(line):
        punctuation = set(p for p in "[].,()=[]")
        for val in punctuation:
            line = line.replace(val, " ")
        flag = line.split(" ")[0]
        return flag

    def non_conforming_flags(self, types=None):
        if types is None:
            types = ARG_TYPES

        nc_list = []
        for ut in self.data:
            for flag in self.data[ut]:
                if self.data[ut][flag] and self.data[ut][flag] not in types:
                    nc_list.append(":".join([ut, flag, self.data[ut][flag]]))
        return nc_list

    def convert_flag_types(self, mapping=None):
        if mapping is None:
            mapping = TYPE_MAPS

        for ut in self.data:
            for flag in self.data[ut]:
                if self.data[ut][flag]:
                    arg_type = self.data[ut][flag]
                    if arg_type == 'n' or 'size' in flag:
                        self.data[ut][flag] = '[Medium Number]'
                    elif 'file' in flag:
                        self.data[ut][flag] = '[File]'
                    else:
                        for t in mapping:
                            for substr in mapping[t]:
                                if substr in self.data[ut][flag].lower():
                                    self.data[ut][flag] = t


if __name__ == "__main__":
    c = []
    with open("command_generator/all_commands.txt") as f:
        for l in f.readlines():
            c.append(l.strip())
            
    scraper = ManScraper(c)
    scraper.extract_utilities()
    scraper.save_json("command_generator/syntax.json", 'command_generator/utility_map.json')