from typing import List
from .rankers import LlmRanker, SearchResult
import openai
import time
import re
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
import copy
from collections import Counter
import tiktoken
import random

random.seed(929)


class SetwiseLlmRanker(LlmRanker):
    CHARACTERS = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
    ]  # "Passage X" and "Passage Y" will be tokenized into 3 tokens, so we dont use for now

    def __init__(self,
                 model_name_or_path,
                 tokenizer_name_or_path,
                 device,
                 num_child=3,
                 k=10,
                 scoring='generation',
                 method="heapsort",
                 num_permutation=1,
                 cache_dir=None,
                 compare_prompt_variant='default'):

        self.device = device
        self.num_child = num_child
        self.num_permutation = num_permutation
        self.k = k
        self.config = AutoConfig.from_pretrained(
            model_name_or_path, cache_dir=cache_dir
        )
        if self.config.model_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(
                (
                    tokenizer_name_or_path
                    if tokenizer_name_or_path is not None
                    else model_name_or_path
                ),
                cache_dir=cache_dir,
            )
            self.llm = T5ForConditionalGeneration.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                cache_dir=cache_dir,
            )
            self.decoder_input_ids = (
                self.tokenizer.encode(
                    "<pad> Passage", return_tensors="pt", add_special_tokens=False
                ).to(self.device)
                if self.tokenizer
                else None
            )

            self.target_token_ids = self.tokenizer.batch_encode_plus(
                [
                    f"<pad> Passage {self.CHARACTERS[i]}"
                    for i in range(len(self.CHARACTERS))
                ],
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            ).input_ids[:, -1]
        elif self.config.model_type == "llama":
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, cache_dir=cache_dir
            )
            self.tokenizer.use_default_system_prompt = False
            if "vicuna" and "v1.5" in model_name_or_path:
                self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                cache_dir=cache_dir,
            ).eval()
        else:
            raise NotImplementedError(
                f"Model type {self.config.model_type} is not supported yet for setwise:("
            )

        self.scoring = scoring
        self.method = method
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.total_insertion_sort_inserts = 0
        self.compare_prompt_variant = compare_prompt_variant

    def compare(self, query: str, docs: List):
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation

        passages = "\n\n".join([f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)])
        if self.compare_prompt_variant == 'default':
            input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                        + passages + '\n\n Output only the passage label of the most relevant passage:'
        else:
            input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                        + passages + '\n\n If their relevance is similar, or none of them is relevant, output A. Output only the passage label of the most relevant passage:'

        if self.scoring == "generation":
            if self.config.model_type == "t5":

                if self.num_permutation == 1:
                    input_ids = self.tokenizer(
                        input_text, return_tensors="pt"
                    ).input_ids.to(self.device)
                    self.total_prompt_tokens += input_ids.shape[1]

                    output_ids = self.llm.generate(
                        input_ids,
                        decoder_input_ids=self.decoder_input_ids,
                        max_new_tokens=2,
                    )[0]

                    self.total_completion_tokens += output_ids.shape[0]

                    output = self.tokenizer.decode(
                        output_ids, skip_special_tokens=True
                    ).strip()
                    output = output[-1]
                else:
                    id_passage = [(i, p) for i, p in enumerate(docs)]
                    labels = [self.CHARACTERS[i] for i in range(len(docs))]
                    batch_data = []
                    for _ in range(self.num_permutation):
                        batch_data.append(
                            [
                                random.sample(id_passage, len(id_passage)),
                                random.sample(labels, len(labels)),
                            ]
                        )

                    batch_ref = []
                    input_text = []
                    for batch in batch_data:
                        ref = []
                        passages = []
                        characters = []
                        for p, c in zip(batch[0], batch[1]):
                            ref.append(p[0])
                            passages.append(p[1].text)
                            characters.append(c)
                        batch_ref.append((ref, characters))
                        passages = "\n\n".join(
                            [
                                f'Passage {characters[i]}: "{passages[i]}"'
                                for i in range(len(passages))
                            ]
                        )
                        input_text.append(
                            f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n'
                            + passages
                            + "\n\nOutput only the passage label of the most relevant passage:"
                        )

                    input_ids = self.tokenizer(
                        input_text, return_tensors="pt"
                    ).input_ids.to(self.device)
                    self.total_prompt_tokens += input_ids.shape[1] * input_ids.shape[0]

                    output_ids = self.llm.generate(
                        input_ids,
                        decoder_input_ids=self.decoder_input_ids.repeat(
                            input_ids.shape[0], 1
                        ),
                        max_new_tokens=2,
                    )
                    output = self.tokenizer.batch_decode(
                        output_ids[:, self.decoder_input_ids.shape[1] :],
                        skip_special_tokens=True,
                    )

                    # vote
                    candidates = []
                    for ref, result in zip(batch_ref, output):
                        result = result.strip().upper()
                        docids, characters = ref
                        if len(result) != 1 or result not in characters:
                            print(f"Unexpected output: {result}")
                            continue
                        win_doc = docids[characters.index(result)]
                        candidates.append(win_doc)

                    if len(candidates) == 0:
                        print(f"Unexpected voting: {output}")
                        output = "Unexpected voting."
                    else:
                        # handle tie
                        candidate_counts = Counter(candidates)
                        max_count = max(candidate_counts.values())
                        most_common_candidates = [
                            candidate
                            for candidate, count in candidate_counts.items()
                            if count == max_count
                        ]
                        if len(most_common_candidates) == 1:
                            output = self.CHARACTERS[most_common_candidates[0]]
                        else:
                            output = self.CHARACTERS[
                                random.choice(most_common_candidates)
                            ]

            elif self.config.model_type == "llama":
                conversation = [{"role": "user", "content": input_text}]

                prompt = self.tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                prompt += " Passage:"

                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                    self.device
                )
                self.total_prompt_tokens += input_ids.shape[1]

                output_ids = self.llm.generate(
                    input_ids,
                    do_sample=False,
                    temperature=0.0,
                    top_p=None,
                    max_new_tokens=1,
                )[0]

                self.total_completion_tokens += output_ids.shape[0]

                output = (
                    self.tokenizer.decode(
                        output_ids[input_ids.shape[1] :], skip_special_tokens=True
                    )
                    .strip()
                    .upper()
                )

        elif self.scoring == "likelihood":
            if self.config.model_type == "t5":
                input_ids = self.tokenizer(
                    input_text, return_tensors="pt"
                ).input_ids.to(self.device)
                self.total_prompt_tokens += input_ids.shape[1]
                with torch.no_grad():
                    logits = self.llm(
                        input_ids=input_ids, decoder_input_ids=self.decoder_input_ids
                    ).logits[0][-1]
                    distributions = torch.softmax(logits, dim=0)
                    scores = distributions[self.target_token_ids[: len(docs)]]
                    ranked = sorted(
                        zip(self.CHARACTERS[: len(docs)], scores),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    output = ranked[0][0]

            else:
                raise NotImplementedError

        if len(output) == 1 and output in self.CHARACTERS:
            pass
        else:
            print(f"Unexpected output: {output}")

        return output

    def heapify(self, arr, n, i, query):
        # Find largest among root and children
        if self.num_child * i + 1 < n:  # if there are children
            docs = [arr[i]] + arr[
                self.num_child * i + 1 : min((self.num_child * (i + 1) + 1), n)
            ]
            inds = [i] + list(
                range(self.num_child * i + 1, min((self.num_child * (i + 1) + 1), n))
            )
            output = self.compare(query, docs)
            try:
                best_ind = self.CHARACTERS.index(output)
            except ValueError:
                best_ind = 0
            try:
                largest = inds[best_ind]
            except IndexError:
                largest = i
            # If root is not largest, swap with largest and continue heapifying
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.heapify(arr, n, largest, query)

    def heapSort(self, arr, query, k):
        n = len(arr)
        ranked = 0
        # Build max heap
        for i in range(n // self.num_child, -1, -1):
            self.heapify(arr, n, i, query)
        for i in range(n - 1, 0, -1):
            # Swap
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            # Heapify root element
            self.heapify(arr, i, 0, query)

    def likelihood_ranking(self, query, docs):
        """
        Sorts docs relevance in a single compare run, using the logits
        """
        self.total_compare += 1
        passages = "\n\n".join([f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)])
        input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                        + passages + '\n\n Output only the passage label of the most relevant passage:'

        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        self.total_prompt_tokens += input_ids.shape[1]

        with torch.no_grad():
            if self.config.model_type == 't5':
                logits = self.llm(input_ids=input_ids, decoder_input_ids=self.decoder_input_ids).logits[0][-1]
            else:
                logits = self.llm()
            distributions = torch.softmax(logits, dim=0)
            scores = distributions[self.target_token_ids[:len(docs)]]
            ranked = sorted(zip([i for i in range(len(docs))], scores), key=lambda x: x[1], reverse=True)
        return ranked
    
    def sort_c_elements(self, query, docs_to_sort):
        """
        Use LLM to sort `c` elements in one call. We can use logits for that if we have an access
        to them
        """
        if self.config.model_type == 't5':
            return [r[0] for r in self.likelihood_ranking(query, docs_to_sort)]
        else:
            # TODO: for black-box LLMs we could ask for a ranking e.g. 1>2>3
            raise NotImplementedError
    
    def bubble_insert(self, sorted_list, el, query, sort_compare: bool = False):
        """
        Inserts element el into a already sorted list (descending order),
        by "bubbling" it up. If `sort_compare` is True, asks model to order elements in
        each comparison. If `sort_compare` is False, just checks if the newly added
        element is the greatest among compared (theoretically less optimal).
        """
        pos_el = len(sorted_list)
        sorted_list = sorted_list + [el]

        # bubble el up
        while 0 < pos_el:
            upper_bound = pos_el + 1
            lower_bound = max(0, upper_bound - self.num_child - 1)

            # assumption here: el is at the end of the to_sort
            to_sort = sorted_list[lower_bound:upper_bound]
            if sort_compare:
                ranking = self.sort_c_elements(query, to_sort)
                
                el_rank = ranking.index(len(to_sort) - 1)
                sorted_to_sort = [to_sort[t] for t in ranking]
            else:
                # we just check which element is the largest out of
                # elements, and move it to the front of the compared elements
                output = self.compare(query, to_sort)
                try:
                    best_ind = self.CHARACTERS.index(output)
                except ValueError:
                    best_ind = 0
                # If the best_ind is too large (hallucination)
                if best_ind >= len(to_sort):
                    best_ind = 0

                if best_ind == len(to_sort) - 1:
                    el_rank = 0
                else:
                    el_rank = len(to_sort) - 1
                sorted_to_sort = [to_sort[best_ind]] + to_sort[:best_ind] + to_sort[(best_ind + 1):]
            
            sorted_list = sorted_list[:lower_bound] + sorted_to_sort + sorted_list[upper_bound:]

            if el_rank != 0:
                # Stop bubbling if el is not the best in to_sort
                # for `sort_compare` = True, this means that the model correctly sorted the array
                # for `sort_compare` = False this means that the element `el` is almost correctly placed
                # (+ - num_children errors)
                break
            else:
                pos_el = lower_bound
        return sorted_list
        

    def insertionSort(self, arr, query, k, sort_compare: bool = False):
        """
        Sorts top_k elements, then scans the rest of the documents. If it finds
        documents more relevant than the smallest one in the top_k, it inserts the document
        into top_k.

        if `sort_compare` is True, uses more efficient logits sorting, if `False` uses
        less efficient max document querying.
        For blackbox models `sort_compare` has to be False.
        """
        # Divide elements into topk, candidates and discarded
        current_top_k_elements = copy.deepcopy(arr[:k])
        candidates = copy.deepcopy(arr[k:])
        discarded = []

        # sort the top_k elements
        self.heapSort(current_top_k_elements, query, k)
        current_top_k_elements = list(reversed(current_top_k_elements))

        # Go through the candidates, and if I find element better than the worst one from the current top k,
        # replace them
        while len(candidates) > 0:
            candidates_to_compare = copy.deepcopy(candidates[:self.num_child])
            candidates = candidates[self.num_child:]

            worst_in_top_k = current_top_k_elements[-1]
            to_compare = [worst_in_top_k, *candidates_to_compare]

            if sort_compare:
                ranking = self.sort_c_elements(query, to_compare)
                best_ind = ranking[0]
            else:
                output = self.compare(query, to_compare)
                ranking = None
                try:
                    best_ind = self.CHARACTERS.index(output)
                except ValueError:
                    best_ind = 0

                # If the best_ind is too large (hallucination)
                if best_ind >= len(to_compare):
                    best_ind = 0

            # if there is a more relevant document than the smallest one in top k
            if best_ind != 0:
                self.total_insertion_sort_inserts += 1

                remaining_to_compare = []

                if sort_compare:
                    # put these that were better than the current worst_in_top_k back in candidates
                    # discard the rest (including worst_in_top_k)
                    next_to_discard = False
                    for i in range(1, len(ranking), 1):
                        if ranking[i] == 0:
                            next_to_discard = True
                        if next_to_discard:
                            discarded.append(to_compare[ranking[i]])
                        else:
                            remaining_to_compare.append(to_compare[ranking[i]])
                else:
                    # discard only worst_in_top_k
                    discarded.append(worst_in_top_k)
                    for i in range(1, len(to_compare), 1):
                        if i != best_ind:
                            remaining_to_compare.append(to_compare[i])

                candidates = remaining_to_compare + candidates

                # Remove the last element from the top_k
                current_top_k_elements = current_top_k_elements[:-1]
                to_put_in_top_k = to_compare[best_ind]
                # fix our top_k elements with bubble insert
                # TODO: theoretically, we could use bubblesort to further optimize this insertion.
                # for k=10 it doesn't matter, but for larger k it might be faster.
                current_top_k_elements = self.bubble_insert(current_top_k_elements, to_put_in_top_k, query, sort_compare)
            
            # if all the other compared documents are worse
            else:
                discarded += candidates_to_compare

        # Build the result
        return current_top_k_elements + discarded


    def rerank(self,  query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.total_insertion_sort_inserts = 0
        
        if self.method == "heapsort":
            self.heapSort(ranking, query, self.k)
            ranking = list(reversed(ranking))
        # New reranking methods:
        elif self.method == "insertion.sort_compare":
            ranking = self.insertionSort(ranking, query, self.k, sort_compare=True)
        elif self.method == "insertion.max_compare":
            ranking = self.insertionSort(ranking, query, self.k, sort_compare=False)
        elif self.method == "bubblesort":
            last_start = len(ranking) - (self.num_child + 1)

            for i in range(self.k):
                start_ind = last_start
                end_ind = last_start + (self.num_child + 1)
                is_change = False
                while True:
                    if start_ind < i:
                        start_ind = i
                    output = self.compare(query, ranking[start_ind:end_ind])
                    try:
                        best_ind = self.CHARACTERS.index(output)
                    except ValueError:
                        best_ind = 0
                    if best_ind != 0:
                        ranking[start_ind], ranking[start_ind + best_ind] = (
                            ranking[start_ind + best_ind],
                            ranking[start_ind],
                        )
                        if not is_change:
                            is_change = True
                            if (
                                last_start != len(ranking) - (self.num_child + 1)
                                and best_ind == len(ranking[start_ind:end_ind]) - 1
                            ):
                                last_start += len(ranking[start_ind:end_ind]) - 1

                    if start_ind == i:
                        break

                    if not is_change:
                        last_start -= self.num_child

                    start_ind -= self.num_child
                    end_ind -= self.num_child

        ##  this is a bit slower but standard bobblesort implementation, keep here FYI
        # elif self.method == "bubblesort":
        #     for i in range(k):
        #         start_ind = len(ranking) - (self.num_child + 1)
        #         end_ind = len(ranking)
        #         while True:
        #             if start_ind < i:
        #                 start_ind = i
        #             output = self.compare(query, ranking[start_ind:end_ind])
        #             try:
        #                 best_ind = self.CHARACTERS.index(output)
        #             except ValueError:
        #                 best_ind = 0
        #             if best_ind != 0:
        #                 ranking[start_ind], ranking[start_ind + best_ind] = ranking[start_ind + best_ind], ranking[start_ind]
        #
        #             if start_ind == i:
        #                 break
        #
        #             start_ind -= self.num_child
        #             end_ind -= self.num_child

        else:
            raise NotImplementedError(f"Method {self.method} is not implemented.")

        results = []
        top_doc_ids = set()
        rank = 1

        for i, doc in enumerate(ranking[: self.k]):
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1

        return results

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.tokenize(text)[:length]
        )


class OpenAiSetwiseLlmRanker(SetwiseLlmRanker):
    def __init__(
        self, model_name_or_path, api_key, num_child=3, method="heapsort", k=10
    ):
        self.llm = model_name_or_path
        self.tokenizer = tiktoken.encoding_for_model(model_name_or_path)
        self.num_child = num_child
        self.method = method
        self.k = k
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.system_prompt = "You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage from a pool of passages based on their relevance to the query."
        openai.api_key = api_key

    def compare(self, query: str, docs: List):
        self.total_compare += 1
        passages = "\n\n".join(
            [
                f'Passage {self.CHARACTERS[i]}: "{doc.text}"'
                for i, doc in enumerate(docs)
            ]
        )
        input_text = (
            f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n'
            + passages
            + "\n\nOutput only the passage label of the most relevant passage."
        )

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.llm,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": input_text},
                    ],
                    temperature=0.0,
                    request_timeout=15,
                )

                self.total_completion_tokens += int(
                    response["usage"]["completion_tokens"]
                )
                self.total_prompt_tokens += int(response["usage"]["prompt_tokens"])

                output = response["choices"][0]["message"]["content"]
                matches = re.findall(r"(Passage [A-Z])", output, re.MULTILINE)
                if matches:
                    output = matches[0][8]
                elif output.strip() in self.CHARACTERS:
                    pass
                else:
                    print(f"Unexpected output: {output}")
                    output = "A"
                return output

            except openai.error.APIError as e:
                # Handle API error here, e.g. retry or log
                print(f"OpenAI API returned an API Error: {e}")
                time.sleep(5)
                continue
            except openai.error.APIConnectionError as e:
                # Handle connection error here
                print(f"Failed to connect to OpenAI API: {e}")
                time.sleep(5)
                continue
            except openai.error.RateLimitError as e:
                # Handle rate limit error (we recommend using exponential backoff)
                print(f"OpenAI API request exceeded rate limit: {e}")
                time.sleep(5)
                continue
            except openai.error.InvalidRequestError as e:
                # Handle invalid request error
                print(f"OpenAI API request was invalid: {e}")
                raise e
            except openai.error.AuthenticationError as e:
                # Handle authentication error
                print(f"OpenAI API request failed authentication: {e}")
                raise e
            except openai.error.Timeout as e:
                # Handle timeout error
                print(f"OpenAI API request timed out: {e}")
                time.sleep(5)
                continue
            except openai.error.ServiceUnavailableError as e:
                # Handle service unavailable error
                print(
                    f"OpenAI API request failed with a service unavailable error: {e}"
                )
                time.sleep(5)
                continue
            except Exception as e:
                print(f"Unknown error: {e}")
                raise e

    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])
