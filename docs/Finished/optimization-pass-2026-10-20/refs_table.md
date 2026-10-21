
#### .claude/skills/autocroissant-analysis-toolkit/SKILL.md (16 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 79 | `psd_analyzer.py:686` | `psd_analyzer.py:688` | blame | self._gap_pattern = re_compile(r'\s{3,}') |
| 127 | `:1307` | `:1309` | symbol | problems.append("ABILITY TEXT NOT FOUND") |
| 127 | `psd_analyzer.py:772` | `psd_analyzer.py:774` | symbol | card.problems.append("NO ABILITY LAYER") |
| 129 | `psd_analyzer.py:1244-1248` | `psd_analyzer.py:1246-1250` | blame | # Append remaining types to the last line |
| 130 | `psd_analyzer.py:1146` | `psd_analyzer.py:1148` | blame | ability_text = sub(r'\s+([:;,\.\?!])', r'\1', ability_text) |
| 148 | `psd_analyzer.py:1827` | `psd_analyzer.py:1829` | symbol | def update_stats(interaction: Optional[Interaction] = None, |
| 287 | `(:1870` | `(:1872` | symbol | if num_new > 0 or traverser.changed: |
| 287 | `psd_analyzer.py:1868` | `psd_analyzer.py:1870` | blame | if num_new > 0: |
| 291 | `psd_analyzer.py:1678` | `psd_analyzer.py:1669` | symbol | num_new += 1 |
| 340 | `psd_analyzer.py:543-545` | `psd_analyzer.py:545-547` | blame | print("Git token found, API limited to 5000 requests/hour.") |
| 362 | `main.py:749-803` | `main.py:762-816` | blame | @tasks.loop(seconds=1) |
| 373 | `main.py:111-114` | `main.py:113-117` | symbol | queue_command(init_reminder) |
| 376 | `main.py:751` | `main.py:764` | symbol | queue_command(check_reminder) |
| 378 | `main.py:759-786` | `main.py:772-799` | blame | async def process_dispatch_queue(): |
| 390 | `main.py:749-803` | `main.py:762-816` | blame | @tasks.loop(seconds=1) |
| 443 | `psd_analyzer.py:1673-1674` | `psd_analyzer.py:1664-1665` | blame | if old.path != path or getattr(old, "blob_sha", None) != sha |

#### .claude/skills/autocroissant-architecture-contract/SKILL.md (47 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 39 | `main.py:750` | `main.py:763` | symbol | async def process_command_queue(): |
| 39 | `main.py:759` | `main.py:772` | symbol | async def process_dispatch_queue(): |
| 40 | `main.py:791` | `main.py:804` | blame | async def process_edit_queue(): |
| 48 | `main.py:750-754` | `main.py:763-767` | symbol | async def process_command_queue(): |
| 49 | `main.py:759-786` | `main.py:772-799` | symbol | async def process_dispatch_queue(): |
| 50 | `main.py:791-803` | `main.py:804-816` | symbol | async def process_edit_queue(): |
| 56 | `main.py:740-745` | `main.py:753-758` | symbol | async def run_command(func, args, kwargs): |
| 59 | `main.py:742-743` | `main.py:755-756` | symbol | if kwargs.get("executor"): |
| 86 | `main.py:778-786` | `main.py:791-799` | symbol | except (HTTPException, AttributeError): |
| 93 | `main.py:101-119` | `main.py:103-122` | symbol | async def on_ready(): |
| 94 | `main.py:111-114` | `main.py:113-117` | symbol | queue_command(init_reminder) |
| 97 | `main.py:204` | `main.py:207` | symbol | bind_reminder_client(client)  # lets the scheduler skip remi |
| 106 | `main.py:116-118` | `main.py:119-121` | manual | process_command_queue.start() |
| 106 | `main.py:145-147` | `main.py:148-150` | symbol | process_command_queue.stop() |
| 106 | `main.py:157-159` | `main.py:160-162` | symbol | process_command_queue.stop() |
| 106 | `main.py:194-196` | `main.py:197-199` | manual | process_command_queue.stop() |
| 108 | `psd_analyzer.py:2288` | `psd_analyzer.py:2295` | symbol | stats_db = StatsDatabase() |
| 108 | `query_card.py:48-51` | `query_card.py:50-53` | blame | if GIT_TOKEN: |
| 108 | `query_card.py:714` | `query_card.py:718` | symbol | card_repo = CardRepository() |
| 108 | `type_emojis.py:26` | `type_emojis.py:30` | symbol | type_emojis: dict[str, str] = {} |
| 109 | `psd_analyzer.py:26` | `psd_analyzer.py:25` | symbol | from commands.query_card import card_repo |
| 109 | `query_card.py:15` | `query_card.py:16` | symbol | from commands.type_emojis import render_type_tokens |
| 110 | `main.py:751` | `main.py:764` | symbol | queue_command(check_reminder) |
| 111 | `main.py:154-156` | `main.py:157-159` | symbol | if perms_check(interaction): |
| 122 | `main.py:256-258` | `main.py:259-261` | symbol | async def slash_query_name(interaction: Interaction, |
| 123 | `main.py:251` | `main.py:254` | symbol | query_name = to_thread(query_name) |
| 132 | `main.py:759` | `main.py:772` | symbol | async def process_dispatch_queue(): |
| 144 | `main.py:363` | `main.py:366` | symbol | await interaction.response.defer() |
| 146 | `psd_analyzer.py:1827` | `psd_analyzer.py:1829` | symbol | def update_stats(interaction: Optional[Interaction] = None, |
| 148 | `psd_analyzer.py:1460-1464` | `psd_analyzer.py:1462-1466` | blame | if not force_update and not self.only and _tree_etags.get(re |
| 152 | `psd_analyzer.py:1379-1391` | `psd_analyzer.py:1381-1393` | blame | def _psd_changed(self, name: str, path: str, sha: Optional[s |
| 155 | `psd_analyzer.py:1429-1440, 1657-1701` | `psd_analyzer.py:1431-1442, 1648-1692` | blame | pool = ThreadPoolExecutor(DOWNLOAD_WORKERS) |
| 160 | `psd_analyzer.py:2288` | `psd_analyzer.py:2295` | symbol | stats_db = StatsDatabase() |
| 180 | `main.py:751` | `main.py:764` | symbol | queue_command(check_reminder) |
| 194 | `main.py:740` | `main.py:753` | symbol | async def run_command(func, args, kwargs): |
| 250 | `psd_analyzer.py:26` | `psd_analyzer.py:25` | blame | from commands.query_card import card_repo |
| 252 | `query_card.py:379` | `query_card.py:383` | symbol | from commands.psd_analyzer import get_stats_as_dict |
| 254 | `query_card.py:15` | `query_card.py:16` | symbol | from commands.type_emojis import render_type_tokens |
| 255 | `type_emojis.py:60` | `type_emojis.py:98` | manual | from commands.query_card import card_repo  # lazy, like quer |
| 285 | `main.py:304` | `main.py:307` | symbol | async def slash_set_ratio(interaction: Interaction, |
| 286 | `main.py:314` | `main.py:317` | symbol | async def slash_set_ratio(interaction: Interaction, |
| 286 | `main.py:597` | `main.py:610` | symbol | async def slash_delete_song(interaction: Interaction, |
| 287 | `main.py:604` | `main.py:617` | symbol | async def slash_delete_song(interaction: Interaction): |
| 297 | `main.py:779` | `main.py:792` | symbol | send_kwargs.pop("ephemeral", None) |
| 302 | `psd_analyzer.py:1667-1668, 1695-1701` | `psd_analyzer.py:1658-1659, 1686-1692` | symbol | finally: |
| 320 | `main.py:165-172` | `main.py:168-175` | blame | git_pull = to_thread(git_pull) |
| 352 | `main.py:742-743` | `main.py:755-756` | symbol | if kwargs.get("executor"): |

#### .claude/skills/autocroissant-build-and-env/SKILL.md (3 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 55 | `commands/psd_analyzer.py:22` | `commands/psd_analyzer.py:21` | blame | import pandas as pd |
| 223 | `psd_analyzer.py:543-545` | `psd_analyzer.py:545-547` | symbol | print("Git token found, API limited to 5000 requests/hour.") |
| 223 | `query_card.py:49-51` | `query_card.py:51-53` | symbol | print("Git token found, API limited to 5000 requests/hour.") |

#### .claude/skills/autocroissant-change-control/SKILL.md (13 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 19 | `global_config.py:5-8` | `global_config.py:5-9` | blame | ALIAS_PKL       = "aliases.pkl" |
| 45 | `main.py:165-198` | `main.py:168-201` | blame | git_pull = to_thread(git_pull) |
| 61 | `main.py:194-196` | `main.py:197-199` | blame | process_command_queue.stop() |
| 83 | `main.py:188` | `main.py:191` | symbol | force_reset="If True, discard all local changes and reset to |
| 154 | `main.py:352-364` | `main.py:355-367` | symbol | update_stats = to_thread(update_stats) |
| 181 | `main.py:178-180` | `main.py:181-183` | blame | if perms_check(interaction): |
| 182 | `main.py:659-672` | `main.py:672-685` | symbol | sync_type_emojis = to_thread(sync_type_emojis) |
| 191 | `help.py:114` | `help.py:118` | symbol | def print_help(interaction: Interaction, help_wanted: str): |
| 204 | `main.py:100-119` | `main.py:102-122` | symbol | @client.event |
| 206 | `main.py:92-96` | `main.py:94-98` | symbol | async def sync_guild_commands(guild): |
| 207 | `main.py:651` | `main.py:664` | symbol | @tree.command(name="sync_global", description="Sync all slas |
| 229 | `:214` | `:216` | symbol | class CardStats: |
| 229 | `psd_analyzer.py:444` | `psd_analyzer.py:446` | symbol | class CardInfo: |

#### .claude/skills/autocroissant-config-and-flags/SKILL.md (21 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 54 | `commands/query_card.py:48-51` | `commands/query_card.py:50-53` | blame | if GIT_TOKEN: |
| 60 | `main.py:811` | `main.py:824` | symbol | client.run(TOKEN) |
| 126 | `main.py:143-149` | `main.py:146-152` | symbol | @tree.command(name="restart_bot", description="Restart the b |
| 175 | `commands/psd_analyzer.py (lines 31-63` | `commands/psd_analyzer.py (lines 33-65` | blame | # ======================== |
| 182 | `psd_analyzer.py:1693` | `psd_analyzer.py:1684` | symbol | if num_updated % UPDATE_RATE == 0: |
| 185 | `psd_analyzer.py:2208, 2215` | `psd_analyzer.py:2215, 2222` | symbol | filename = f"{EXPORTED_STATS_NAME}.txt" |
| 186 | `psd_analyzer.py:2269` | `psd_analyzer.py:2276` | symbol | filename = f"{EXPORTED_RULES_NAME}.txt" |
| 187 | `psd_analyzer.py:2065, 2150, 2199` | `psd_analyzer.py:2072, 2157, 2206` | symbol | existing_columns = [col for col in COLUMN_ORDER if col in df |
| 198 | `psd_analyzer.py:1255-1261` | `psd_analyzer.py:1257-1263` | symbol | EXCESSIVE_STAT_EXCLUSIONS: set[str] = { |
| 199 | `psd_analyzer.py:1262-1281` | `psd_analyzer.py:1264-1283` | symbol | ABILITY_EXCLUSIONS: set[str] = { |
| 206 | `commands/query_card.py (lines 19-24` | `commands/query_card.py (lines 21-26` | blame | # ======================== |
| 214 | `query_card.py:714` | `query_card.py:718` | blame | card_repo = CardRepository() |
| 246 | `commands/type_emojis.py (lines 14-26` | `commands/type_emojis.py (lines 18-30` | symbol | # ======================== |
| 293 | `setup.py (lines 8-45` | `setup.py (lines 8-44` | blame | # Files that should NOT be compiled (will cause issues) |
| 319 | `query_card.py:670-687` | `query_card.py:674-691` | symbol | async def set_match_ratio(interaction: Interaction, value: O |
| 320 | `query_card.py:690-708` | `query_card.py:694-712` | symbol | async def set_repository(interaction: Interaction, new_repo: |
| 335 | `main.py:314` | `main.py:317` | symbol | async def slash_set_ratio(interaction: Interaction, |
| 343 | `main.py:361` | `main.py:364` | blame | force_update: Optional[bool] = False, |
| 343 | `psd_analyzer.py:1829` | `psd_analyzer.py:1831` | blame | force_update: bool = False, |
| 423 | `main.py:361` | `main.py:364` | symbol | force_update: Optional[bool] = False, |
| 423 | `psd_analyzer.py:1829` | `psd_analyzer.py:1831` | symbol | force_update: bool = False, |

#### .claude/skills/autocroissant-debugging-playbook/SKILL.md (23 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 34 | `commands/psd_analyzer.py:1827` | `commands/psd_analyzer.py:1829` | symbol | def update_stats(interaction: Optional[Interaction] = None, |
| 44 | `psd_analyzer.py:1673` | `psd_analyzer.py:1664` | blame | if old.path != path or getattr(old, "blob_sha", None) != sha |
| 45 | `psd_analyzer.py:581` | `psd_analyzer.py:583` | symbol | def prune_clean_cards(self) -> None: |
| 47 | `psd_analyzer.py:1888-1889` | `psd_analyzer.py:1895-1896` | symbol | queue_command(card_repo.populate_files, traverser.tree) |
| 100 | `main.py:758-786` | `main.py:771-799` | blame | @tasks.loop(seconds=1) |
| 107 | `main.py:779` | `main.py:792` | blame | send_kwargs.pop("ephemeral", None) |
| 111 | `main.py:790-803` | `main.py:803-816` | blame | @tasks.loop(seconds=1) |
| 120 | `main.py:143-149` | `main.py:146-152` | symbol | @tree.command(name="restart_bot", description="Restart the b |
| 121 | `main.py:194-196` | `main.py:197-199` | blame | process_command_queue.stop() |
| 123 | `main.py:779` | `main.py:792` | symbol | send_kwargs.pop("ephemeral", None) |
| 124 | `main.py:796-803` | `main.py:809-816` | blame | except (HTTPException, AttributeError): |
| 124 | `psd_analyzer.py:1731` | `psd_analyzer.py:1722` | symbol | def _send_progress(interaction: Optional[Interaction], count |
| 124 | `psd_analyzer.py:34` | `psd_analyzer.py:36` | symbol | UPDATE_RATE = 25 |
| 131 | `main.py:145-147, 157-159, 194-196` | `main.py:148-150, 160-162, 197-199` | blame/symbol | process_command_queue.stop() |
| 141 | `main.py:92-109` | `main.py:94-111` | blame | async def sync_guild_commands(guild): |
| 152 | `main.py:304` | `main.py:307` | symbol | async def slash_set_ratio(interaction: Interaction, |
| 152 | `main.py:597` | `main.py:610` | symbol | async def slash_delete_song(interaction: Interaction, |
| 179 | `main.py:810-816` | `main.py:823-829` | symbol | try: |
| 180 | `main.py:87` | `main.py:89` | symbol | intents.message_content = True |
| 182 | `psd_analyzer.py:554-579` | `psd_analyzer.py:556-581` | symbol | def load(self) -> None: |
| 229 | `main.py:194-196` | `main.py:197-199` | blame | process_command_queue.stop() |
| 234 | `main.py:166` | `main.py:169` | symbol | @tree.command(name="pull", description="Does a hard reset, t |
| 242 | `main.py:751` | `main.py:764` | blame | queue_command(check_reminder) |

#### .claude/skills/autocroissant-docs-and-style/SKILL.md (39 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 77 | `help.py:114-134` | `help.py:118-138` | symbol | def print_help(interaction: Interaction, help_wanted: str): |
| 79 | `help.py:122-123` | `help.py:126-127` | symbol | elif help_wanted == "stats": |
| 80 | `help.py:124-125` | `help.py:128-129` | symbol | else: |
| 83 | `main.py:132-136` | `main.py:135-139` | blame | Choice(name="text", value="text"), |
| 90 | `main.py:131-137` | `main.py:134-140` | symbol | @app_commands.choices(help_type=[ |
| 122 | `main.py:439` | `main.py:452` | symbol | @tree.command(name="view_old_metadata", description="View ol |
| 123 | `main.py:676` | `main.py:689` | symbol | @tree.command(name="list_guild_members", description="Lists  |
| 124 | `main.py:691` | `main.py:704` | symbol | @tree.command(name="list_guild_channels", description="Lists |
| 125 | `main.py:706` | `main.py:719` | symbol | @tree.command(name="get_channel_messages", description="Get  |
| 129 | `main.py:132-136` | `main.py:135-139` | blame | Choice(name="text", value="text"), |
| 141 | `main.py:166` | `main.py:169` | blame | @tree.command(name="pull", description="Does a hard reset, t |
| 150 | `main.py:166` | `main.py:169` | blame | @tree.command(name="pull", description="Does a hard reset, t |
| 153 | `help.py:76` | `help.py:80` | manual | "- only_ability: Export only ability text (default: True)\n" |
| 155 | `main.py:374` | `main.py:387` | symbol | only_ability: Optional[bool] = False, |
| 156 | `main.py:370` | `main.py:383` | blame | only_ability='Should we only output the ability of cards?', |
| 168 | `main.py:660-672` | `main.py:673-687` | manual | @tree.command(name="sync_emojis", description="Uploads missi |
| 173 | `main.py:335` | `main.py:338` | symbol | blend_mode='How to blend: slice, or average. Default = slice |
| 173 | `main.py:615` | `main.py:628` | symbol | num='The number of messages to delete. Default = 100.', |
| 174 | `main.py:543` | `main.py:556` | symbol | multiplier='The multiplier of the volume. 0.5 = half as loud |
| 175 | `main.py:627` | `main.py:640` | symbol | @tree.command(name="list_guilds", description="Lists all gui |
| 176 | `main.py:131-137` | `main.py:134-140` | symbol | @app_commands.choices(help_type=[ |
| 176 | `main.py:337-340` | `main.py:340-343` | symbol | @app_commands.choices(blend_mode=[ |
| 176 | `main.py:459-464` | `main.py:472-477` | symbol | @app_commands.choices(field=[ |
| 178 | `main.py:302` | `main.py:305` | symbol | value='If empty, posts the current tolerance. Otherwise sets |
| 178 | `main.py:312` | `main.py:315` | symbol | new_repo='If empty, posts the current repository. Otherwise  |
| 179 | `main.py:640, 680, 713-714` | `main.py:653, 693, 726-727` | symbol | guild_id: str, |
| 199 | `psd_analyzer.py:29` | `psd_analyzer.py:31` | blame | getLogger("psd_tools").setLevel(CRITICAL) |
| 202 | `query_card.py:25` | `query_card.py:26` | blame | GIT_TOKEN: str = getattr(config, "GIT_TOKEN", "") |
| 209 | `psd_analyzer.py:2288-2289` | `psd_analyzer.py:2295-2296` | symbol | stats_db = StatsDatabase() |
| 210 | `query_card.py:714-715` | `query_card.py:718-719` | symbol | card_repo = CardRepository() |
| 220 | `main.py:123-125` | `main.py:126-128` | blame | #################### |
| 247 | `query_card.py:61-69` | `query_card.py:63-71` | symbol | """ |
| 269 | `help.py:129` | `help.py:133` | symbol | help_msg += f"{cmd}\n{desc}\n\n" |
| 270 | `main.py:96` | `main.py:98` | symbol | print(f"Synced commands to guild: {guild.name} ({guild.id})" |
| 278 | `help.py:114-134` | `help.py:118-138` | symbol | def print_help(interaction: Interaction, help_wanted: str): |
| 287 | `query_card.py:25` | `query_card.py:26` | symbol | GIT_TOKEN: str = getattr(config, "GIT_TOKEN", "") |
| 483 | `main.py:166` | `main.py:169` | blame | @tree.command(name="pull", description="Does a hard reset, t |
| 488 | `help.py:76` | `help.py:80` | symbol | "- only_ability: Export only ability text (default: True)\n" |
| 488 | `main.py:374` | `main.py:387` | symbol | only_ability: Optional[bool] = False, |

#### .claude/skills/autocroissant-failure-archaeology/SKILL.md (7 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 132 | `psd_analyzer.py:686` | `psd_analyzer.py:688` | symbol | self._gap_pattern = re_compile(r'\s{3,}') |
| 156 | `psd_analyzer.py:1146` | `psd_analyzer.py:1148` | symbol | ability_text = sub(r'\s+([:;,\.\?!])', r'\1', ability_text) |
| 233 | `psd_analyzer.py:1244-1248` | `psd_analyzer.py:1246-1250` | blame | # Append remaining types to the last line |
| 301 | `psd_analyzer.py:1262` | `psd_analyzer.py:1264` | symbol | ABILITY_EXCLUSIONS: set[str] = { |
| 306 | `psd_analyzer.py:841-844` | `psd_analyzer.py:843-846` | symbol | elif (layer.name.lower() in self.all_types and layer.is_visi |
| 307 | `psd_analyzer.py:50` | `psd_analyzer.py:52` | symbol | MISSPELT_CARD_TYPES = ['undread', 'tornado', 'error'] |
| 399 | `main.py:361` | `main.py:364` | symbol | force_update: Optional[bool] = False, |

#### .claude/skills/autocroissant-psd-extraction-campaign/SKILL.md (16 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 42 | `psd_analyzer.py:686` | `psd_analyzer.py:688` | symbol | self._gap_pattern = re_compile(r'\s{3,}') |
| 43 | `psd_analyzer.py:841` | `psd_analyzer.py:843` | symbol | elif (layer.name.lower() in self.all_types and layer.is_visi |
| 45 | `psd_analyzer.py:38,743` | `psd_analyzer.py:40,745` | symbol | card_mid_y = int(psd.height * TYPE_REGION_RATIO) |
| 46 | `psd_analyzer.py:1206` | `psd_analyzer.py:1208` | symbol | def _inject_type_names(self, ability: str, type_bboxes: list |
| 47 | `psd_analyzer.py:1244-1248` | `psd_analyzer.py:1246-1250` | blame | # Append remaining types to the last line |
| 304 | `psd_analyzer.py:1146` | `psd_analyzer.py:1148` | symbol | ability_text = sub(r'\s+([:;,\.\?!])', r'\1', ability_text) |
| 408 | `psd_analyzer.py:1146` | `psd_analyzer.py:1148` | blame | ability_text = sub(r'\s+([:;,\.\?!])', r'\1', ability_text) |
| 483 | `psd_analyzer.py:1305` | `psd_analyzer.py:1307` | symbol | not card.ability and |
| 531 | `main.py:388` | `main.py:401` | symbol | @tree.command(name="update_metadata", description="Edit meta |
| 531 | `psd_analyzer.py:1947` | `psd_analyzer.py:1954` | symbol | def manual_metadata_entry(interaction: Interaction, |
| 536 | `psd_analyzer.py:1997` | `psd_analyzer.py:2004` | symbol | stats_db.stats[name] = CardInfo( |
| 537 | `psd_analyzer.py:1379` | `psd_analyzer.py:1381` | symbol | def _psd_changed(self, name: str, path: str, sha: Optional[s |
| 580 | `main.py:361` | `main.py:364` | symbol | force_update: Optional[bool] = False, |
| 582 | `psd_analyzer.py:1829` | `psd_analyzer.py:1831` | symbol | force_update: bool = False, |
| 584 | `psd_analyzer.py:1495` | `psd_analyzer.py:1498` | symbol | def _populate_types_from_local(self, local_path: str) -> Non |
| 608 | `main.py:388` | `main.py:401` | symbol | @tree.command(name="update_metadata", description="Edit meta |

#### .claude/skills/autocroissant-research-frontier/SKILL.md (14 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 43 | `commands/query_card.py:266` | `commands/query_card.py:287` | symbol | def rulebook_search_engine(self, search_text: str) -> DataFr |
| 44 | `main.py:285` | `main.py:289` | blame | @tree.command(name="query_rulebook", description="Search the |
| 44 | `query_card.py:472` | `query_card.py:564` | blame | def query_rulebook(interaction: Interaction, |
| 46 | `query_card.py:194` | `query_card.py:215` | symbol | def ability_search_engine(self, ability: str) -> DataFrame: |
| 57 | `commands/psd_analyzer.py:1718` | `commands/psd_analyzer.py:2269` | def-lookup | def export_rulebook_to_file(interaction: Interaction) -> Non |
| 57 | `main.py:382` | `main.py:394` | symbol | @tree.command(name="export_rulebook", description="Exports t |
| 58 | `main.py:369` | `main.py:381` | symbol | @tree.command(name="export_cards", description="Exports all  |
| 58 | `psd_analyzer.py:1623` | `psd_analyzer.py:2174` | def-lookup | def export_stats_to_file(interaction: Interaction, |
| 59 | `psd_analyzer.py:593` | `psd_analyzer.py:1208` | symbol | def _inject_type_names(self, ability: str, type_bboxes: list |
| 141 | `main.py:495` | `main.py:508` | blame | await play_music(interaction, song, play_next) |
| 218 | `psd_analyzer.py:1206-1250` | `psd_analyzer.py:1208-1252` | symbol | def _inject_type_names(self, ability: str, type_bboxes: list |
| 282 | `commands/psd_analyzer.py:123` | `commands/psd_analyzer.py:446` | symbol | class CardInfo: |
| 292 | `psd_analyzer.py:138` | `psd_analyzer.py:472` | symbol | def to_dict(self) -> dict: |
| 293 | `psd_analyzer.py:210` | `psd_analyzer.py:549` | symbol | def save(self) -> None: |

#### .claude/skills/autocroissant-research-methodology/SKILL.md (7 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 155 | `psd_analyzer.py:1870` | `psd_analyzer.py:1872` | symbol | if num_new > 0 or traverser.changed: |
| 196 | `psd_analyzer.py:282` | `psd_analyzer.py:621` | blame | "MDW": lambda: {"type": CardType.MDW.value, "ability": None} |
| 196 | `psd_analyzer.py:31` | `psd_analyzer.py:41` | symbol | EXCLUDE_FOLDERS = ["Markers", "MDW"] |
| 231 | `psd_analyzer.py:42` | `psd_analyzer.py:44` | blame | GIT_TOKEN: str = getattr(config, "GIT_TOKEN", "") |
| 232 | `query_card.py:25` | `query_card.py:26` | blame | GIT_TOKEN: str = getattr(config, "GIT_TOKEN", "") |
| 232 | `type_emojis.py:17` | `type_emojis.py:21` | manual | GIT_TOKEN: str = getattr(config, "GIT_TOKEN", "") |
| 267 | `analytics.py:209` | `analytics.py:426` | manual | func = slash_registry.get(name) |

#### .claude/skills/autocroissant-run-and-operate/SKILL.md (32 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 32 | `main.py:9` | `main.py:11` | blame | from config import TOKEN |
| 46 | `psd_analyzer.py:542-545` | `psd_analyzer.py:544-547` | blame | if GIT_TOKEN: |
| 47 | `query_card.py:48-51` | `query_card.py:50-53` | blame | if GIT_TOKEN: |
| 48 | `main.py:102` | `main.py:104` | symbol | print(f'We have logged in as {client.user}') |
| 50 | `main.py:116-119` | `main.py:119-122` | blame | process_command_queue.start() |
| 51 | `main.py:96` | `main.py:98` | blame | print(f"Synced commands to guild: {guild.name} ({guild.id})" |
| 67 | `main.py:101-119` | `main.py:103-122` | symbol | async def on_ready(): |
| 71 | `main.py:116-118` | `main.py:119-121` | blame | process_command_queue.start() |
| 77 | `psd_analyzer.py:2289` | `psd_analyzer.py:2296` | symbol | def init_psd() -> None: |
| 78 | `query_card.py:715` | `query_card.py:719` | symbol | def init_query() -> None: |
| 79 | `type_emojis.py:218` | `type_emojis.py:389` | symbol | async def init_type_emojis(client: Client) -> None: |
| 83 | `main.py:749-803` | `main.py:762-816` | blame | @tasks.loop(seconds=1) |
| 102 | `main.py:152-162` | `main.py:155-165` | symbol | @tree.command(name="stop_bot", description="Stop the bot.") |
| 103 | `main.py:143-149` | `main.py:146-152` | symbol | @tree.command(name="restart_bot", description="Restart the b |
| 104 | `main.py:812-813` | `main.py:825-826` | blame | except KeyboardInterrupt: |
| 129 | `main.py:92-96` | `main.py:94-98` | symbol | async def sync_guild_commands(guild): |
| 133 | `main.py:651-656` | `main.py:664-669` | symbol | @tree.command(name="sync_global", description="Sync all slas |
| 172 | `main.py:166` | `main.py:169` | blame | @tree.command(name="pull", description="Does a hard reset, t |
| 194 | `main.py:186-199` | `main.py:189-202` | blame | @tree.command(name="update", description="Push pickles, pull |
| 218 | `main.py:194-196` | `main.py:197-199` | blame | process_command_queue.stop() |
| 265 | `main.py:352-364` | `main.py:355-367` | blame | update_stats = to_thread(update_stats) |
| 265 | `psd_analyzer.py:1827` | `psd_analyzer.py:1829` | blame | def update_stats(interaction: Optional[Interaction] = None, |
| 291 | `main.py:360-362` | `main.py:363-365` | blame | output_problematic_cards: Optional[bool] = True, |
| 302 | `psd_analyzer.py:1745` | `psd_analyzer.py:1736` | blame | msg = f"{num_new} cards changed or new, {num_old} unchanged, |
| 302 | `psd_analyzer.py:34,1693` | `psd_analyzer.py:36,1684` | blame | if num_updated % UPDATE_RATE == 0: |
| 309 | `psd_analyzer.py:1868-1871` | `psd_analyzer.py:1870-1873` | blame | if num_new > 0: |
| 327 | `psd_analyzer.py:1410-1440` | `psd_analyzer.py:1412-1442` | symbol | def _blob(self, sha: str, stream: bool = False): |
| 377 | `psd_analyzer.py:1429-1440` | `psd_analyzer.py:1431-1442` | symbol | def _download(self, path: str, sha: str) -> Optional[str]: |
| 385 | `global_config.py:10` | `global_config.py:14` | symbol | TIMEZONE        = ZoneInfo("America/Los_Angeles") |
| 386 | `main.py:751` | `main.py:764` | symbol | queue_command(check_reminder) |
| 390 | `main.py:206` | `main.py:209` | symbol | @tree.command(name="set_reminder", description="Set a remind |
| 470 | `main.py:612-623` | `main.py:625-636` | symbol | @tree.command(name="purge", description="Deletes messages.") |

#### .claude/skills/autocroissant-validation-and-qa/SKILL.md (25 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 51 | `psd_analyzer.py:831` | `psd_analyzer.py:1257` | manual | EXCESSIVE_STAT_EXCLUSIONS: set[str] = { |
| 51 | `psd_analyzer.py:838` | `psd_analyzer.py:1264` | manual | ABILITY_EXCLUSIONS: set[str] = { |
| 63 | `psd_analyzer.py:199` | `psd_analyzer.py:538` | symbol | self.stats: dict[str, CardInfo] = {} |
| 65 | `psd_analyzer.py:487` | `psd_analyzer.py:670` | blame | elif len(folders) > 2: |
| 66 | `psd_analyzer.py:521` | `psd_analyzer.py:704` | symbol | existing_card = self.stats_db.stats.get(name) |
| 67 | `psd_analyzer.py:45` | `psd_analyzer.py:52` | symbol | MISSPELT_CARD_TYPES = ['undread', 'tornado', 'error'] |
| 69 | `psd_analyzer.py:595` | `psd_analyzer.py:781` | blame |  |
| 71 | `psd_analyzer.py:628-660` | `psd_analyzer.py:814-846` | blame | shape = layer.numpy("shape") |
| 76 | `psd_analyzer.py:230` | `psd_analyzer.py:500` | manual | result["problem"] = self.problems |
| 80 | `psd_analyzer.py:863-898` | `psd_analyzer.py:1051-1086` | symbol | Stars drawn on a card whose star count comes from the PSD. |
| 82 | `psd_analyzer.py:900` | `psd_analyzer.py:1325` | def-lookup | def _validate_stats(card: CardInfo) -> list[str]: |
| 84 | `psd_analyzer.py:1213` | `psd_analyzer.py:1744` | manual | def _format_problems( |
| 85 | `psd_analyzer.py:1165` | `psd_analyzer.py:1695` | manual | return (self._format_problems(problematic_cards), num_new +  |
| 90 | `psd_analyzer.py:892-894` | `psd_analyzer.py:1317-1319` | manual | if ("NO ABILITY LAYER" in card.problems and |
| 92 | `psd_analyzer.py:1155` | `psd_analyzer.py:1351` | blame | return problems |
| 156 | `psd_analyzer.py:379-380, 929` | `psd_analyzer.py:719-720, 1649` | blame | downloads = [pool.submit(self._download, path, sha) for path |
| 196 | `global_config.py:6-7` | `global_config.py:6-8` | blame | STATS_PKL       = "stats.pkl" |
| 234 | `psd_analyzer.py:1827` | `psd_analyzer.py:1829` | blame | def update_stats(interaction: Optional[Interaction] = None, |
| 261 | `psd_analyzer.py:1379` | `psd_analyzer.py:1381` | symbol | def _psd_changed(self, name: str, path: str, sha: Optional[s |
| 263 | `psd_analyzer.py:1719` | `psd_analyzer.py:1710` | symbol | commits = repo.get_commits(path=path) |
| 271 | `psd_analyzer.py:1888-1889` | `psd_analyzer.py:1895-1896` | symbol | queue_command(card_repo.populate_files, traverser.tree) |
| 381 | `psd_analyzer.py:670` | `psd_analyzer.py:1285` | symbol | @staticmethod |
| 382 | `psd_analyzer.py:707` | `psd_analyzer.py:1324` | symbol | @staticmethod |
| 386 | `psd_analyzer.py:456` | `psd_analyzer.py:856` | blame | card.problems.append(f"MISSPELT TYPE: {layer.name.lower()}") |
| 481 | `main.py:361` | `main.py:364` | symbol | force_update: Optional[bool] = False, |

#### .claude/skills/impossibility-cards-reference/SKILL.md (42 refs)

| line | as written | replace with | check | code now at the new line |
|---|---|---|---|---|
| 38 | `commands/psd_analyzer.py:361` | `commands/psd_analyzer.py:701` | blame | name = splitext(basename(relative_path))[0].replace('_', ' ' |
| 44 | `commands/psd_analyzer.py:256-340` | `commands/psd_analyzer.py:595-679` | symbol | class CardClassifier: |
| 68 | `psd_analyzer.py:285, 290, 309, 335` | `psd_analyzer.py:624, 629, 648, 674` | blame | "stars": int(folders[len(folders) - 1].split()[0]), |
| 113 | `psd_analyzer.py:39` | `psd_analyzer.py:41` | symbol | EXCLUDE_FOLDERS = ["Markers", "MDW"] |
| 114 | `psd_analyzer.py:1648, 1568` | `psd_analyzer.py:1639, 1571` | blame | if (not m or any(folder in EXCLUDE_FOLDERS for folder in pat |
| 127 | `psd_analyzer.py:428-456` | `psd_analyzer.py:829-856` | symbol | def _process_layer(self, layer, card: CardInfo, |
| 157 | `psd_analyzer.py:434` | `psd_analyzer.py:835` | blame | if layer.kind == "type": |
| 165 | `psd_analyzer.py:469, 407` | `psd_analyzer.py:869, 747` | manual | is_rulepage = "Rulebook" in relative_path |
| 166 | `psd_analyzer.py:460-467` | `psd_analyzer.py:868-889` | manual | engine_text = layer.engine_dict["Editor"]["Text"] |
| 179 | `psd_analyzer.py:529-539` | `psd_analyzer.py:1139-1149` | blame | def _process_abilities(self, abilities: list[tuple[str, Boun |
| 180 | `psd_analyzer.py:422` | `psd_analyzer.py:774` | blame | card.problems.append("NO ABILITY LAYER") |
| 187 | `psd_analyzer.py:1038` | `psd_analyzer.py:1498` | def-lookup | def _populate_types_from_local(self, local_path: str) -> Non |
| 194 | `psd_analyzer.py:442-447` | `psd_analyzer.py:843-848` | blame | elif (layer.name.lower() in self.all_types and layer.is_visi |
| 196 | `psd_analyzer.py:30, 405` | `psd_analyzer.py:40, 745` | blame | card_mid_y = int(psd.height * TYPE_REGION_RATIO) |
| 205 | `psd_analyzer.py:473-515` | `psd_analyzer.py:997-1047` | blame | def _is_stat_layer(self, layer) -> bool: |
| 238 | `psd_analyzer.py:556-583` | `psd_analyzer.py:1171-1198` | symbol | @staticmethod |
| 248 | `psd_analyzer.py:593-637` | `psd_analyzer.py:1208-1252` | symbol | def _inject_type_names(self, ability: str, type_bboxes: list |
| 254 | `psd_analyzer.py:30` | `psd_analyzer.py:40` | symbol | TYPE_REGION_RATIO = 0.5 |
| 255 | `psd_analyzer.py:348` | `psd_analyzer.py:688` | blame | self._gap_pattern = re_compile(r'\s{3,}') |
| 256 | `psd_analyzer.py:557` | `psd_analyzer.py:1172` | blame | def _sort_by_position(items: list[tuple[str, BoundingBox]],  |
| 257 | `psd_analyzer.py:590` | `psd_analyzer.py:1205` | symbol | max_height = max(bboxes[len(bboxes) - 1][1].y // 3, card_mid |
| 258 | `psd_analyzer.py:538` | `psd_analyzer.py:1148` | blame | ability_text = sub(r'\s+([:;,\.\?!])', r'\1', ability_text) |
| 273 | `psd_analyzer.py:632-635` | `psd_analyzer.py:1247-1250` | blame | if type_index < len(types) and result_lines: |
| 400 | `psd_analyzer.py:640-733` | `psd_analyzer.py:1255-1351` | symbol | class CardValidator: |
| 405 | `psd_analyzer.py:422` | `psd_analyzer.py:774` | blame | card.problems.append("NO ABILITY LAYER") |
| 405 | `psd_analyzer.py:455-456` | `psd_analyzer.py:855-856` | symbol | elif (layer.name.lower() in MISSPELT_CARD_TYPES and layer.is |
| 426 | `psd_analyzer.py:642-668` | `psd_analyzer.py:1257-1283` | blame | EXCESSIVE_STAT_EXCLUSIONS: set[str] = { |
| 465 | `psd_analyzer.py:44-48` | `psd_analyzer.py:46-50` | symbol | COLUMN_ORDER = [ |
| 470 | `psd_analyzer.py:1706` | `psd_analyzer.py:1697` | blame | def _get_remote_timestamp(self, |
| 475 | `main.py:361` | `main.py:364` | symbol | force_update: Optional[bool] = False, |
| 475 | `psd_analyzer.py:1827` | `psd_analyzer.py:1829` | symbol | def update_stats(interaction: Optional[Interaction] = None, |
| 506 | `psd_analyzer.py:1673-1674` | `psd_analyzer.py:1664-1665` | blame | if old.path != path or getattr(old, "blob_sha", None) != sha |
| 507 | `psd_analyzer.py:581-590` | `psd_analyzer.py:583-592` | symbol | def prune_clean_cards(self) -> None: |
| 509 | `psd_analyzer.py:1367-1377` | `psd_analyzer.py:1369-1379` | symbol | def _update_old_stats_paths(self) -> None: |
| 519 | `:626` | `:718` | symbol | card_repo = CardRepository() |
| 519 | `commands/query_card.py:27` | `commands/query_card.py:29` | symbol | @dataclass |
| 526 | `query_card.py:23` | `query_card.py:25` | symbol | DEFAULT_MATCH_RATIO = 0.6 |
| 538 | `query_card.py:526-579` | `query_card.py:618-671` | manual | async def manage_alias(interaction: Interaction, key: Option |
| 543 | `query_card.py:194-264` | `query_card.py:215-285` | symbol | def ability_search_engine(self, ability: str) -> DataFrame: |
| 577 | `query_card.py:257-260` | `query_card.py:278-281` | symbol | scores = self.cards_dff["ability"].apply( |
| 582 | `query_card.py:266-290` | `query_card.py:287-311` | symbol | def rulebook_search_engine(self, search_text: str) -> DataFr |
| 584 | `main.py:268-281` | `main.py:272-285` | symbol | @tree.command(name="query_ability", description="Query a car |
