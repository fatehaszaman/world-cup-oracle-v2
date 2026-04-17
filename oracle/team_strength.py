"""
oracle/team_strength.py — Composite team strength scorer.

BUSINESS SUMMARY
----------------
This module answers one question: "How good is this national team, overall?"
It combines five independent signals — squad market value, how strong each
position is (with real player names and ratings), the country's football
infrastructure, historical World Cup results, and commercial/sponsorship
power — into a single number between 0 and 1. That number drives every
match probability in the Monte Carlo simulation.

DEVELOPER NOTES
---------------
Architecture:
  TeamStrengthScorer orchestrates five sub-scorers, each independently
  normalized to [0, 1], then blended via configurable dimension weights
  (config.DIMENSION_WEIGHTS). Every sub-scorer has a hardcoded fallback
  data dictionary so the system runs without any API keys.

Complexity:
  - score_all_teams(): O(T × P) where T=teams, P=6 positions. Negligible.
  - All sub-scores: O(1) per team lookup.

Data freshness: Squad values and player ratings reflect 2025/26 season
  assessments. Ratings are calibrated to FIFA 25 / Sofascore scale (0–99).
"""

from __future__ import annotations

import math
import logging
from typing import Optional

from config import (
    DIMENSION_WEIGHTS,
    POSITION_WEIGHTS,
    HISTORICAL_POINTS,
    HISTORICAL_MAX_POINTS,
    RESOURCE_WEIGHTS,
    GDP_NORMALIZATION_CEILING,
    POPULATION_LOG_MIN,
    POPULATION_LOG_MAX,
    SQUAD_VALUE_CEILING,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded squad market values (EUR millions) — 2025/26 estimates
# Source: Transfermarkt methodology adapted for national squads
# ---------------------------------------------------------------------------
SQUAD_MARKET_VALUES_EUR_M: dict[str, float] = {
    "England":       1_100.0,
    "France":        1_050.0,
    "Brazil":          900.0,
    "Germany":         980.0,
    "Spain":           870.0,
    "Portugal":        780.0,
    "Argentina":       850.0,
    "Netherlands":     760.0,
    "Belgium":         680.0,
    "Italy":           620.0,
    "Croatia":         350.0,
    "Uruguay":         310.0,
    "Mexico":          280.0,
    "Colombia":        270.0,
    "Senegal":         240.0,
    "United States":   250.0,
    "Morocco":         280.0,
    "Japan":           220.0,
    "South Korea":     200.0,
    "Switzerland":     330.0,
    "Denmark":         340.0,
    "Austria":         280.0,
    "Poland":          210.0,
    "Serbia":          190.0,
    "Ecuador":         140.0,
    "Canada":          150.0,
    "Australia":       130.0,
    "Nigeria":         170.0,
    "Ivory Coast":     180.0,
    "Cameroon":        120.0,
    "Saudi Arabia":     90.0,
    "Iran":             70.0,
}

# ---------------------------------------------------------------------------
# Positional ratings with named players (2025/26 assessments)
# Format per position: {"rating": float, "starter": "Name (rating)", "backup": "Name (rating)"}
# Ratings calibrated to Sofascore/FIFA 25 scale (0–100)
# ---------------------------------------------------------------------------
POSITIONAL_DATA: dict[str, dict[str, dict]] = {
    "France": {
        "GK": {"rating": 87, "starter": "Mike Maignan (87)", "backup": "Alphonse Areola (82)"},
        "CB": {"rating": 86, "starter": "William Saliba (88)", "backup": "Dayot Upamecano (84)"},
        "FB": {"rating": 90, "starter": "Theo Hernandez (90)", "backup": "Jules Koundé (87)"},
        "CM": {"rating": 85, "starter": "Aurélien Tchouaméni (86)", "backup": "Adrien Rabiot (82)"},
        "AM": {"rating": 93, "starter": "Kylian Mbappé (95)", "backup": "Antoine Griezmann (87)"},
        "FW": {"rating": 88, "starter": "Ousmane Dembélé (87)", "backup": "Marcus Thuram (86)"},
    },
    "England": {
        "GK": {"rating": 85, "starter": "Jordan Pickford (84)", "backup": "Dean Henderson (80)"},
        "CB": {"rating": 85, "starter": "Harry Maguire (82)", "backup": "John Stones (84)"},
        "FB": {"rating": 88, "starter": "Trent Alexander-Arnold (89)", "backup": "Luke Shaw (82)"},
        "CM": {"rating": 86, "starter": "Declan Rice (87)", "backup": "Jude Bellingham (90)"},
        "AM": {"rating": 84, "starter": "Phil Foden (87)", "backup": "Jack Grealish (82)"},
        "FW": {"rating": 88, "starter": "Harry Kane (90)", "backup": "Bukayo Saka (86)"},
    },
    "Brazil": {
        "GK": {"rating": 88, "starter": "Alisson (91)", "backup": "Ederson (87)"},
        "CB": {"rating": 86, "starter": "Marquinhos (87)", "backup": "Éder Militão (85)"},
        "FB": {"rating": 91, "starter": "Trent Alexander-Arnold (89)", "backup": "Danilo (82)"},
        "CM": {"rating": 85, "starter": "Casemiro (88)", "backup": "Bruno Guimarães (86)"},
        "AM": {"rating": 90, "starter": "Vinícius Júnior (92)", "backup": "Rodrygo (87)"},
        "FW": {"rating": 87, "starter": "Raphinha (86)", "backup": "Gabriel Jesus (82)"},
    },
    "Germany": {
        "GK": {"rating": 86, "starter": "Manuel Neuer (85)", "backup": "Marc-André ter Stegen (85)"},
        "CB": {"rating": 82, "starter": "Antonio Rüdiger (85)", "backup": "Jonathan Tah (83)"},
        "FB": {"rating": 84, "starter": "Joshua Kimmich (87)", "backup": "David Raum (81)"},
        "CM": {"rating": 88, "starter": "Toni Kroos (87)", "backup": "Florian Wirtz (88)"},
        "AM": {"rating": 86, "starter": "Jamal Musiala (89)", "backup": "Leroy Sané (84)"},
        "FW": {"rating": 83, "starter": "Kai Havertz (82)", "backup": "Thomas Müller (79)"},
    },
    "Spain": {
        "GK": {"rating": 84, "starter": "Unai Simón (83)", "backup": "David Raya (83)"},
        "CB": {"rating": 84, "starter": "Aymeric Laporte (82)", "backup": "Robin Le Normand (81)"},
        "FB": {"rating": 86, "starter": "Alejandro Grimaldo (85)", "backup": "Dani Carvajal (84)"},
        "CM": {"rating": 92, "starter": "Pedri (90)", "backup": "Rodrigo (Rodri) (93)"},
        "AM": {"rating": 87, "starter": "Gavi (86)", "backup": "Fabián Ruiz (84)"},
        "FW": {"rating": 85, "starter": "Lamine Yamal (87)", "backup": "Dani Olmo (83)"},
    },
    "Argentina": {
        "GK": {"rating": 82, "starter": "Emiliano Martínez (88)", "backup": "Franco Armani (80)"},
        "CB": {"rating": 81, "starter": "Cristian Romero (85)", "backup": "Lisandro Martínez (84)"},
        "FB": {"rating": 83, "starter": "Nahuel Molina (81)", "backup": "Nicolás Tagliafico (79)"},
        "CM": {"rating": 84, "starter": "Rodrigo De Paul (83)", "backup": "Enzo Fernández (83)"},
        "AM": {"rating": 95, "starter": "Lionel Messi (93)", "backup": "Alexis Mac Allister (82)"},
        "FW": {"rating": 87, "starter": "Lautaro Martínez (86)", "backup": "Julián Álvarez (84)"},
    },
    "Portugal": {
        "GK": {"rating": 80, "starter": "Diogo Costa (83)", "backup": "Rui Patrício (79)"},
        "CB": {"rating": 82, "starter": "Rúben Dias (88)", "backup": "Pepe (80)"},
        "FB": {"rating": 84, "starter": "Nuno Mendes (83)", "backup": "João Cancelo (82)"},
        "CM": {"rating": 83, "starter": "Bernardo Silva (87)", "backup": "Rúben Neves (82)"},
        "AM": {"rating": 87, "starter": "Bruno Fernandes (85)", "backup": "João Félix (81)"},
        "FW": {"rating": 90, "starter": "Cristiano Ronaldo (85)", "backup": "Rafael Leão (83)"},
    },
    "Netherlands": {
        "GK": {"rating": 81, "starter": "Bart Verbruggen (80)", "backup": "Jasper Cillessen (77)"},
        "CB": {"rating": 83, "starter": "Virgil van Dijk (87)", "backup": "Stefan de Vrij (82)"},
        "FB": {"rating": 81, "starter": "Denzel Dumfries (82)", "backup": "Daley Blind (76)"},
        "CM": {"rating": 83, "starter": "Frenkie de Jong (85)", "backup": "Tijjani Reijnders (81)"},
        "AM": {"rating": 84, "starter": "Memphis Depay (81)", "backup": "Xavi Simons (82)"},
        "FW": {"rating": 87, "starter": "Cody Gakpo (83)", "backup": "Wout Weghorst (79)"},
    },
    "Belgium": {
        "GK": {"rating": 80, "starter": "Koen Casteels (80)", "backup": "Simon Mignolet (78)"},
        "CB": {"rating": 80, "starter": "Wout Faes (80)", "backup": "Arthur Theate (78)"},
        "FB": {"rating": 80, "starter": "Timothy Castagne (79)", "backup": "Thomas Meunier (76)"},
        "CM": {"rating": 82, "starter": "Youri Tielemans (82)", "backup": "Axel Witsel (77)"},
        "AM": {"rating": 86, "starter": "Kevin De Bruyne (88)", "backup": "Dries Mertens (79)"},
        "FW": {"rating": 86, "starter": "Romelu Lukaku (84)", "backup": "Leandro Trossard (81)"},
    },
    "Italy": {
        "GK": {"rating": 85, "starter": "Gianluigi Donnarumma (87)", "backup": "Alex Meret (82)"},
        "CB": {"rating": 84, "starter": "Alessandro Bastoni (86)", "backup": "Giorgio Scalvini (81)"},
        "FB": {"rating": 82, "starter": "Giovanni Di Lorenzo (82)", "backup": "Federico Dimarco (81)"},
        "CM": {"rating": 84, "starter": "Nicolo Barella (86)", "backup": "Sandro Tonali (82)"},
        "AM": {"rating": 79, "starter": "Lorenzo Pellegrini (79)", "backup": "Marco Verratti (80)"},
        "FW": {"rating": 77, "starter": "Gianluca Scamacca (79)", "backup": "Ciro Immobile (78)"},
    },
    "Croatia": {
        "GK": {"rating": 77, "starter": "Dominik Livaković (82)", "backup": "Ivica Ivušić (74)"},
        "CB": {"rating": 76, "starter": "Joško Gvardiol (84)", "backup": "Duje Ćaleta-Car (76)"},
        "FB": {"rating": 75, "starter": "Josip Juranović (77)", "backup": "Borna Sosa (74)"},
        "CM": {"rating": 87, "starter": "Luka Modrić (85)", "backup": "Mateo Kovačić (84)"},
        "AM": {"rating": 80, "starter": "Nikola Vlašić (79)", "backup": "Marcelo Brozović (82)"},
        "FW": {"rating": 75, "starter": "Ante Budimir (74)", "backup": "Ivan Perišić (77)"},
    },
    "Uruguay": {
        "GK": {"rating": 76, "starter": "Fernando Muslera (76)", "backup": "Sergio Rochet (74)"},
        "CB": {"rating": 80, "starter": "José María Giménez (82)", "backup": "Ronald Araújo (83)"},
        "FB": {"rating": 74, "starter": "Nahitan Nández (74)", "backup": "Matías Viña (72)"},
        "CM": {"rating": 78, "starter": "Federico Valverde (86)", "backup": "Matías Vecino (75)"},
        "AM": {"rating": 77, "starter": "Rodrigo Bentancur (78)", "backup": "Giorgian De Arrascaeta (79)"},
        "FW": {"rating": 82, "starter": "Darwin Núñez (82)", "backup": "Luis Suárez (77)"},
    },
    "Mexico": {
        "GK": {"rating": 75, "starter": "Guillermo Ochoa (76)", "backup": "Alfredo Talavera (72)"},
        "CB": {"rating": 73, "starter": "César Montes (73)", "backup": "Héctor Moreno (72)"},
        "FB": {"rating": 75, "starter": "Jorge Sánchez (74)", "backup": "Gerardo Arteaga (73)"},
        "CM": {"rating": 76, "starter": "Edson Álvarez (77)", "backup": "Héctor Herrera (74)"},
        "AM": {"rating": 75, "starter": "Hirving Lozano (77)", "backup": "Andrés Guardado (73)"},
        "FW": {"rating": 74, "starter": "Raúl Jiménez (75)", "backup": "Henry Martín (73)"},
    },
    "Colombia": {
        "GK": {"rating": 74, "starter": "Camilo Vargas (74)", "backup": "David Ospina (73)"},
        "CB": {"rating": 73, "starter": "Davinson Sánchez (76)", "backup": "Yerry Mina (74)"},
        "FB": {"rating": 74, "starter": "Santiago Arias (72)", "backup": "Johan Mojica (72)"},
        "CM": {"rating": 77, "starter": "Wilmar Barrios (76)", "backup": "Jefferson Lerma (74)"},
        "AM": {"rating": 78, "starter": "James Rodríguez (80)", "backup": "Juan Cuadrado (76)"},
        "FW": {"rating": 77, "starter": "Luis Díaz (82)", "backup": "Duván Zapata (77)"},
    },
    "Senegal": {
        "GK": {"rating": 73, "starter": "Édouard Mendy (78)", "backup": "Alfred Gomis (73)"},
        "CB": {"rating": 74, "starter": "Kalidou Koulibaly (83)", "backup": "Abdou Diallo (76)"},
        "FB": {"rating": 72, "starter": "Moussa Wagué (72)", "backup": "Formose Mendy (70)"},
        "CM": {"rating": 74, "starter": "Idrissa Gana Gueye (77)", "backup": "Pape Matar Sarr (76)"},
        "AM": {"rating": 75, "starter": "Ismaïla Sarr (77)", "backup": "Nampalys Mendy (71)"},
        "FW": {"rating": 80, "starter": "Sadio Mané (82)", "backup": "Boulaye Dia (76)"},
    },
    "Morocco": {
        "GK": {"rating": 78, "starter": "Yassine Bounou (84)", "backup": "Munir Mohamedi (74)"},
        "CB": {"rating": 79, "starter": "Romain Saïss (78)", "backup": "Nayef Aguerd (78)"},
        "FB": {"rating": 75, "starter": "Noussair Mazraoui (79)", "backup": "Achraf Hakimi (85)"},
        "CM": {"rating": 76, "starter": "Sofyan Amrabat (80)", "backup": "Azzedine Ounahi (74)"},
        "AM": {"rating": 74, "starter": "Hakim Ziyech (78)", "backup": "Ilias Chair (72)"},
        "FW": {"rating": 73, "starter": "Youssef En-Nesyri (76)", "backup": "Ayoub El Kaabi (73)"},
    },
    "United States": {
        "GK": {"rating": 73, "starter": "Matt Turner (74)", "backup": "Ethan Horvath (71)"},
        "CB": {"rating": 72, "starter": "Miles Robinson (72)", "backup": "Cameron Carter-Vickers (71)"},
        "FB": {"rating": 76, "starter": "Sergino Dest (76)", "backup": "Antonee Robinson (74)"},
        "CM": {"rating": 74, "starter": "Weston McKennie (76)", "backup": "Tyler Adams (76)"},
        "AM": {"rating": 73, "starter": "Christian Pulisic (79)", "backup": "Giovanni Reyna (73)"},
        "FW": {"rating": 74, "starter": "Ricardo Pepi (74)", "backup": "Josh Sargent (73)"},
    },
    "Japan": {
        "GK": {"rating": 72, "starter": "Shuichi Gonda (72)", "backup": "Zion Suzuki (71)"},
        "CB": {"rating": 73, "starter": "Maya Yoshida (73)", "backup": "Ko Itakura (73)"},
        "FB": {"rating": 74, "starter": "Hiroki Sakai (73)", "backup": "Yuto Nagatomo (72)"},
        "CM": {"rating": 77, "starter": "Wataru Endo (77)", "backup": "Hidemasa Morita (75)"},
        "AM": {"rating": 75, "starter": "Takumi Minamino (76)", "backup": "Junya Ito (74)"},
        "FW": {"rating": 73, "starter": "Ao Tanaka (74)", "backup": "Kaoru Mitoma (78)"},
    },
    "South Korea": {
        "GK": {"rating": 71, "starter": "Kim Seung-gyu (71)", "backup": "Jo Hyeon-woo (70)"},
        "CB": {"rating": 71, "starter": "Kim Min-jae (84)", "backup": "Jung Seung-hyun (70)"},
        "FB": {"rating": 73, "starter": "Kim Jin-su (72)", "backup": "Lee Yong (71)"},
        "CM": {"rating": 73, "starter": "Jung Woo-young (72)", "backup": "Hwang In-beom (72)"},
        "AM": {"rating": 76, "starter": "Son Heung-min (83)", "backup": "Lee Kang-in (77)"},
        "FW": {"rating": 76, "starter": "Hwang Hee-chan (76)", "backup": "Cho Gue-sung (72)"},
    },
    "Switzerland": {
        "GK": {"rating": 80, "starter": "Yann Sommer (82)", "backup": "Gregor Kobel (82)"},
        "CB": {"rating": 79, "starter": "Nico Elvedi (79)", "backup": "Manuel Akanji (83)"},
        "FB": {"rating": 77, "starter": "Silvan Widmer (76)", "backup": "Ricardo Rodríguez (75)"},
        "CM": {"rating": 78, "starter": "Granit Xhaka (81)", "backup": "Remo Freuler (78)"},
        "AM": {"rating": 77, "starter": "Xherdan Shaqiri (78)", "backup": "Fabian Frei (74)"},
        "FW": {"rating": 74, "starter": "Haris Seferović (73)", "backup": "Noah Okafor (75)"},
    },
    "Denmark": {
        "GK": {"rating": 79, "starter": "Kasper Schmeichel (79)", "backup": "Oliver Christensen (75)"},
        "CB": {"rating": 78, "starter": "Simon Kjær (79)", "backup": "Joachim Andersen (80)"},
        "FB": {"rating": 76, "starter": "Joakim Mæhle (77)", "backup": "Daniel Wass (74)"},
        "CM": {"rating": 79, "starter": "Christian Eriksen (83)", "backup": "Thomas Delaney (77)"},
        "AM": {"rating": 76, "starter": "Martin Braithwaite (74)", "backup": "Andreas Skov Olsen (75)"},
        "FW": {"rating": 75, "starter": "Kasper Dolberg (74)", "backup": "Jonas Wind (75)"},
    },
    "Austria": {
        "GK": {"rating": 76, "starter": "Patrick Pentz (75)", "backup": "Daniel Bachmann (74)"},
        "CB": {"rating": 75, "starter": "David Alaba (84)", "backup": "Phillipp Lienhart (74)"},
        "FB": {"rating": 75, "starter": "Stefan Lainer (74)", "backup": "Phillipp Mwene (72)"},
        "CM": {"rating": 77, "starter": "Marcel Sabitzer (78)", "backup": "Konrad Laimer (79)"},
        "AM": {"rating": 76, "starter": "Marko Arnautović (77)", "backup": "Christoph Baumgartner (77)"},
        "FW": {"rating": 75, "starter": "Michael Gregoritsch (74)", "backup": "Sasa Kalajdzic (73)"},
    },
    "Poland": {
        "GK": {"rating": 74, "starter": "Wojciech Szczęsny (82)", "backup": "Łukasz Skorupski (75)"},
        "CB": {"rating": 73, "starter": "Jan Bednarek (74)", "backup": "Kamil Glik (73)"},
        "FB": {"rating": 72, "starter": "Matty Cash (74)", "backup": "Bartosz Bereszyński (72)"},
        "CM": {"rating": 73, "starter": "Piotr Zieliński (81)", "backup": "Mateusz Klich (73)"},
        "AM": {"rating": 72, "starter": "Sebastian Szymański (74)", "backup": "Kamil Grosicki (71)"},
        "FW": {"rating": 79, "starter": "Robert Lewandowski (84)", "backup": "Arkadiusz Milik (76)"},
    },
    "Serbia": {
        "GK": {"rating": 72, "starter": "Vanja Milinković-Savić (76)", "backup": "Predrag Rajković (74)"},
        "CB": {"rating": 73, "starter": "Nikola Milenković (77)", "backup": "Stefan Mitrović (72)"},
        "FB": {"rating": 71, "starter": "Strahinja Pavlović (76)", "backup": "Darko Lazović (71)"},
        "CM": {"rating": 73, "starter": "Sergej Milinković-Savić (83)", "backup": "Nemanja Gudelj (73)"},
        "AM": {"rating": 73, "starter": "Dušan Tadić (77)", "backup": "Filip Kostić (76)"},
        "FW": {"rating": 76, "starter": "Dušan Vlahović (83)", "backup": "Aleksandar Mitrović (80)"},
    },
    "Ecuador": {
        "GK": {"rating": 70, "starter": "Alexander Domínguez (71)", "backup": "Hernán Galíndez (69)"},
        "CB": {"rating": 72, "starter": "Piero Hincapié (76)", "backup": "William Pacho (73)"},
        "FB": {"rating": 70, "starter": "Pervis Estupiñán (77)", "backup": "Angelo Preciado (71)"},
        "CM": {"rating": 71, "starter": "Moisés Caicedo (79)", "backup": "Carlos Gruezo (70)"},
        "AM": {"rating": 70, "starter": "Gonzalo Plata (73)", "backup": "Ángel Mena (70)"},
        "FW": {"rating": 73, "starter": "Enner Valencia (75)", "backup": "Michael Estrada (71)"},
    },
    "Canada": {
        "GK": {"rating": 72, "starter": "Maxime Crépeau (72)", "backup": "Milan Borjan (73)"},
        "CB": {"rating": 70, "starter": "Kamal Miller (71)", "backup": "Alistair Johnston (72)"},
        "FB": {"rating": 73, "starter": "Alphonso Davies (83)", "backup": "Richie Laryea (71)"},
        "CM": {"rating": 72, "starter": "Stephen Eustáquio (73)", "backup": "Samuel Piette (70)"},
        "AM": {"rating": 71, "starter": "Jonathan David (79)", "backup": "Tajon Buchanan (74)"},
        "FW": {"rating": 73, "starter": "Cyle Larin (73)", "backup": "Lucas Cavallini (70)"},
    },
    "Australia": {
        "GK": {"rating": 71, "starter": "Mat Ryan (74)", "backup": "Danny Vukovic (70)"},
        "CB": {"rating": 70, "starter": "Harry Souttar (73)", "backup": "Trent Sainsbury (70)"},
        "FB": {"rating": 70, "starter": "Nathaniel Atkinson (70)", "backup": "Joel King (69)"},
        "CM": {"rating": 71, "starter": "Aaron Mooy (74)", "backup": "Jackson Irvine (72)"},
        "AM": {"rating": 70, "starter": "Martin Boyle (71)", "backup": "Craig Goodwin (70)"},
        "FW": {"rating": 72, "starter": "Mitchell Duke (71)", "backup": "Jamie Maclaren (70)"},
    },
    "Nigeria": {
        "GK": {"rating": 70, "starter": "Francis Uzoho (71)", "backup": "Stanley Nwabali (70)"},
        "CB": {"rating": 71, "starter": "William Troost-Ekong (72)", "backup": "Chidozie Awaziem (70)"},
        "FB": {"rating": 70, "starter": "Ola Aina (72)", "backup": "Kenneth Omeruo (69)"},
        "CM": {"rating": 71, "starter": "Frank Onyeka (72)", "backup": "Alex Iwobi (73)"},
        "AM": {"rating": 73, "starter": "Samuel Chukwueze (74)", "backup": "Kelechi Iheanacho (72)"},
        "FW": {"rating": 76, "starter": "Victor Osimhen (83)", "backup": "Moses Simon (73)"},
    },
    "Ivory Coast": {
        "GK": {"rating": 69, "starter": "Badra Ali Sangaré (70)", "backup": "Sylvain Gbohouo (68)"},
        "CB": {"rating": 70, "starter": "Eric Bailly (72)", "backup": "Simon Deli (68)"},
        "FB": {"rating": 70, "starter": "Serge Aurier (72)", "backup": "Ghislain Konan (69)"},
        "CM": {"rating": 71, "starter": "Jean Michaël Seri (72)", "backup": "Franck Kessié (75)"},
        "AM": {"rating": 72, "starter": "Sébastien Haller (73)", "backup": "Wilfried Zaha (73)"},
        "FW": {"rating": 74, "starter": "Nicolas Pépé (73)", "backup": "Jonathan Kodjia (70)"},
    },
    "Cameroon": {
        "GK": {"rating": 68, "starter": "André Onana (81)", "backup": "Simon Ngapandouetnbu (67)"},
        "CB": {"rating": 68, "starter": "Olivier Mbaizo (67)", "backup": "Ambroise Oyongo (67)"},
        "FB": {"rating": 68, "starter": "Collins Fai (68)", "backup": "Nouhou Tolo (67)"},
        "CM": {"rating": 69, "starter": "Pierre Kunde (68)", "backup": "Martin Hongla (68)"},
        "AM": {"rating": 70, "starter": "Bryan Mbeumo (73)", "backup": "Nicolas Moumi Ngamaleu (68)"},
        "FW": {"rating": 72, "starter": "Vincent Aboubakar (73)", "backup": "Eric Maxim Choupo-Moting (72)"},
    },
    "Saudi Arabia": {
        "GK": {"rating": 66, "starter": "Mohammed Al-Owais (68)", "backup": "Abdullah Al-Mayouf (63)"},
        "CB": {"rating": 66, "starter": "Ali Al-Bulayhi (66)", "backup": "Hassan Tambakti (65)"},
        "FB": {"rating": 65, "starter": "Sultan Al-Ghanam (64)", "backup": "Yasser Al-Shahrani (65)"},
        "CM": {"rating": 67, "starter": "Salman Al-Faraj (67)", "backup": "Sami Al-Naji (65)"},
        "AM": {"rating": 67, "starter": "Salem Al-Dawsari (70)", "backup": "Firas Al-Buraikan (67)"},
        "FW": {"rating": 68, "starter": "Saleh Al-Shehri (67)", "backup": "Mohammed Al-Qasim (64)"},
    },
    "Iran": {
        "GK": {"rating": 65, "starter": "Alireza Beiranvand (72)", "backup": "Hossein Hosseini (63)"},
        "CB": {"rating": 65, "starter": "Morteza Pouraliganji (66)", "backup": "Majid Hosseini (64)"},
        "FB": {"rating": 64, "starter": "Ehsan Hajsafi (66)", "backup": "Sadegh Moharrami (63)"},
        "CM": {"rating": 65, "starter": "Saeid Ezatolahi (65)", "backup": "Ahmad Noorollahi (63)"},
        "AM": {"rating": 64, "starter": "Ali Gholizadeh (65)", "backup": "Sardar Azmoun (73)"},
        "FW": {"rating": 65, "starter": "Mehdi Taremi (77)", "backup": "Kaveh Rezaei (63)"},
    },
}

# For backward compatibility — flat ratings dict used by monte_carlo
POSITIONAL_RATINGS: dict[str, dict[str, float]] = {
    team: {pos: data["rating"] for pos, data in pos_data.items()}
    for team, pos_data in POSITIONAL_DATA.items()
}

# ---------------------------------------------------------------------------
# Historical World Cup results (last 5 tournaments: 2006–2022)
# ---------------------------------------------------------------------------
HISTORICAL_RESULTS: dict[str, list[str]] = {
    "France":         ["quarter_finalist", "round_of_16",   "group_stage",     "runner_up",      "winner"],
    "Brazil":         ["quarter_finalist", "runner_up",     "semi_finalist",   "quarter_finalist","quarter_finalist"],
    "Germany":        ["semi_finalist",    "winner",        "winner",          "group_stage",    "group_stage"],
    "Spain":          ["round_of_16",      "winner",        "round_of_16",     "round_of_16",    "round_of_16"],
    "Argentina":      ["quarter_finalist", "quarter_finalist","runner_up",      "round_of_16",    "winner"],
    "England":        ["quarter_finalist", "round_of_16",   "round_of_16",     "quarter_finalist","quarter_finalist"],
    "Italy":          ["runner_up",        "round_of_16",   "group_stage",     "group_stage",    "round_of_16"],
    "Portugal":       ["semi_finalist",    "round_of_16",   "round_of_16",     "quarter_finalist","round_of_16"],
    "Netherlands":    ["round_of_16",      "runner_up",     "semi_finalist",   "round_of_16",    "quarter_finalist"],
    "Croatia":        ["round_of_16",      "round_of_16",   "round_of_16",     "runner_up",      "semi_finalist"],
    "Uruguay":        ["group_stage",      "semi_finalist", "round_of_16",     "quarter_finalist","group_stage"],
    "Belgium":        ["round_of_16",      "round_of_16",   "round_of_16",     "semi_finalist",  "quarter_finalist"],
    "Switzerland":    ["round_of_16",      "round_of_16",   "round_of_16",     "round_of_16",    "quarter_finalist"],
    "Denmark":        ["round_of_16",      "group_stage",   "group_stage",     "round_of_16",    "round_of_16"],
    "Mexico":         ["round_of_16",      "round_of_16",   "round_of_16",     "round_of_16",    "round_of_16"],
    "Colombia":       ["group_stage",      "round_of_16",   "quarter_finalist","group_stage",    "group_stage"],
    "Senegal":        ["group_stage",      "group_stage",   "group_stage",     "round_of_16",    "quarter_finalist"],
    "Morocco":        ["group_stage",      "group_stage",   "group_stage",     "group_stage",    "semi_finalist"],
    "Japan":          ["round_of_16",      "round_of_16",   "round_of_16",     "round_of_16",    "round_of_16"],
    "Poland":         ["group_stage",      "group_stage",   "group_stage",     "round_of_16",    "round_of_16"],
    "Serbia":         ["group_stage",      "group_stage",   "group_stage",     "group_stage",    "group_stage"],
    "Austria":        ["round_of_16",      "group_stage",   "group_stage",     "group_stage",    "group_stage"],
    "Ecuador":        ["group_stage",      "group_stage",   "group_stage",     "group_stage",    "group_stage"],
    "United States":  ["round_of_16",      "round_of_16",   "group_stage",     "group_stage",    "round_of_16"],
    "Canada":         ["group_stage",      "group_stage",   "group_stage",     "group_stage",    "group_stage"],
    "Australia":      ["semi_finalist",    "group_stage",   "group_stage",     "group_stage",    "round_of_16"],
    "South Korea":    ["semi_finalist",    "round_of_16",   "round_of_16",     "group_stage",    "round_of_16"],
    "Nigeria":        ["round_of_16",      "group_stage",   "round_of_16",     "round_of_16",    "group_stage"],
    "Ivory Coast":    ["group_stage",      "group_stage",   "group_stage",     "group_stage",    "group_stage"],
    "Cameroon":       ["group_stage",      "group_stage",   "group_stage",     "group_stage",    "group_stage"],
    "Saudi Arabia":   ["round_of_16",      "group_stage",   "group_stage",     "group_stage",    "round_of_16"],
    "Iran":           ["group_stage",      "group_stage",   "group_stage",     "group_stage",    "group_stage"],
}

FOOTBALL_PRIMARY_SPORT: dict[str, bool] = {
    "France": True, "England": True, "Brazil": True, "Germany": True,
    "Spain": True, "Portugal": True, "Argentina": True, "Netherlands": True,
    "Belgium": True, "Italy": True, "Croatia": True, "Uruguay": True,
    "Mexico": True, "Colombia": True, "Senegal": True, "Morocco": True,
    "Japan": False, "South Korea": False, "Switzerland": True, "Denmark": True,
    "Austria": True, "Poland": True, "Serbia": True, "Ecuador": True,
    "Canada": False, "Australia": False, "Nigeria": True, "Ivory Coast": True,
    "Cameroon": True, "Saudi Arabia": True, "Iran": True,
    "United States": False,
}

FIFA_CONFEDERATION_BUDGET: dict[str, float] = {
    "France": 38.0, "England": 38.0, "Brazil": 30.0, "Germany": 38.0,
    "Spain": 38.0, "Portugal": 38.0, "Argentina": 30.0, "Netherlands": 38.0,
    "Belgium": 38.0, "Italy": 38.0, "Croatia": 38.0, "Switzerland": 38.0,
    "Denmark": 38.0, "Austria": 38.0, "Poland": 38.0, "Serbia": 38.0,
    "Mexico": 30.0, "Colombia": 30.0, "Ecuador": 30.0, "Canada": 41.0,
    "United States": 41.0, "Uruguay": 30.0,
    "Senegal": 25.0, "Morocco": 25.0, "Nigeria": 25.0,
    "Ivory Coast": 25.0, "Cameroon": 25.0,
    "Japan": 45.0, "South Korea": 45.0, "Australia": 45.0,
    "Saudi Arabia": 45.0, "Iran": 45.0,
}


class TeamStrengthScorer:
    """
    Computes a composite [0, 1] strength score for each national team across
    five independently normalised signal dimensions.

    Parameters
    ----------
    custom_weights : dict, optional
        Override config.DIMENSION_WEIGHTS. Must sum to 1.0.

    Methods
    -------
    score_squad_value(team)          → float [0, 1]
    score_positional_power(team)     → float [0, 1]
    score_country_resources(...)     → float [0, 1]
    score_historical_performance(t.) → float [0, 1]
    score_commercial_signal(team)    → float [0, 1]
    composite_score(team, ...)       → float [0, 1]
    score_all_teams(wb_data)         → dict[str, float]
    """

    def __init__(self, custom_weights: Optional[dict[str, float]] = None) -> None:
        self.weights = custom_weights or dict(DIMENSION_WEIGHTS)
        assert abs(sum(self.weights.values()) - 1.0) < 1e-9, \
            "Dimension weights must sum to 1.0"

    # ------------------------------------------------------------------
    # 1. Squad market value
    # ------------------------------------------------------------------

    def score_squad_value(self, team: str) -> float:
        """
        Normalise squad market value (EUR millions) to [0, 1] via linear
        scaling against SQUAD_VALUE_CEILING (config.py).

        Parameters
        ----------
        team : str  Team name matching SQUAD_MARKET_VALUES_EUR_M keys.

        Returns
        -------
        float  Normalised squad value in [0, 1].
        """
        value_eur_m = SQUAD_MARKET_VALUES_EUR_M.get(team)
        if value_eur_m is None:
            logger.warning("No squad value data for '%s'; defaulting to 0.30.", team)
            return 0.30
        return min(value_eur_m / SQUAD_VALUE_CEILING, 1.0)

    # ------------------------------------------------------------------
    # 2. Positional power
    # ------------------------------------------------------------------

    def score_positional_power(self, team: str) -> float:
        """
        Weighted sum of six position-group ratings, normalised to [0, 1].

        Position group ratings are on a 0–100 scale; weights come from
        config.POSITION_WEIGHTS (GK=0.15, CB=0.20, FB=0.10, CM=0.25,
        AM=0.15, FW=0.15). Named player data is stored in POSITIONAL_DATA.

        Parameters
        ----------
        team : str

        Returns
        -------
        float  Weighted positional power in [0, 1].
        """
        pos_data = POSITIONAL_DATA.get(team)
        if pos_data is None:
            logger.warning("No positional data for '%s'; defaulting to 0.55.", team)
            return 0.55

        weighted_sum = sum(
            pos_data.get(pos, {}).get("rating", 65.0) * weight
            for pos, weight in POSITION_WEIGHTS.items()
        )
        return weighted_sum / 100.0

    def get_player_at_position(self, team: str, position: str, role: str = "starter") -> str:
        """
        Return the named player at a given position.

        Parameters
        ----------
        team : str
        position : str  One of GK, CB, FB, CM, AM, FW.
        role : str      "starter" or "backup".

        Returns
        -------
        str  Player name and rating, e.g. "Alisson (91)".
        """
        data = POSITIONAL_DATA.get(team, {}).get(position, {})
        return data.get(role, "Unknown")

    # ------------------------------------------------------------------
    # 3. Country football resources
    # ------------------------------------------------------------------

    def score_country_resources(
        self,
        team: str,
        gdp_per_capita: float,
        population: int,
        fifa_budget: Optional[float] = None,
    ) -> float:
        """
        Infrastructure & resource score combining macroeconomic and football
        governance indicators.

        Formula
        -------
        score = w_gdp   × norm_gdp_per_capita
              + w_pop   × norm_log10_population
              + w_budget× norm_confederation_budget
              + w_sport × football_primary_sport_flag

        Parameters
        ----------
        team : str
        gdp_per_capita : float  GDP per capita, current USD (World Bank).
        population : int        Total population (World Bank).
        fifa_budget : float, optional  Confederation annual budget (EUR M).

        Returns
        -------
        float  Resource score in [0, 1].
        """
        norm_gdp = min(gdp_per_capita / GDP_NORMALIZATION_CEILING, 1.0)

        log_pop = math.log10(max(population, 1_000_000))
        norm_pop = (log_pop - POPULATION_LOG_MIN) / (POPULATION_LOG_MAX - POPULATION_LOG_MIN)
        norm_pop = max(0.0, min(norm_pop, 1.0))

        budget = fifa_budget or FIFA_CONFEDERATION_BUDGET.get(team, 25.0)
        norm_budget = min(budget / 50.0, 1.0)

        sport_bonus = 1.0 if FOOTBALL_PRIMARY_SPORT.get(team, True) else 0.5

        score = (
            RESOURCE_WEIGHTS["gdp_per_capita"] * norm_gdp
            + RESOURCE_WEIGHTS["population"] * norm_pop
            + RESOURCE_WEIGHTS["fifa_budget"] * norm_budget
            + RESOURCE_WEIGHTS["primary_sport"] * sport_bonus
        )
        return max(0.0, min(score, 1.0))

    # ------------------------------------------------------------------
    # 4. Historical performance
    # ------------------------------------------------------------------

    def score_historical_performance(self, team: str) -> float:
        """
        Points-based score from last 5 World Cups (2006–2022), normalised
        to [0, 1] against maximum possible points (5 wins × 3 pts = 15).

        Scoring table: winner=3, runner_up=2, semi=1, quarter=0.5, R16=0.1.

        Parameters
        ----------
        team : str

        Returns
        -------
        float  Historical performance score in [0, 1].
        """
        history = HISTORICAL_RESULTS.get(team)
        if history is None:
            return 0.0

        total_pts = sum(HISTORICAL_POINTS.get(r, 0.0) for r in history)
        return min(total_pts / HISTORICAL_MAX_POINTS, 1.0)

    # ------------------------------------------------------------------
    # 5. Commercial signal
    # ------------------------------------------------------------------

    def score_commercial_signal(self, team: str) -> float:
        """
        Sponsorship / commercial power as a talent-investment proxy.

        Delegates to SponsorshipValuator. Falls back to a squad-value
        proportional estimate if the module is unavailable.

        Parameters
        ----------
        team : str

        Returns
        -------
        float  Commercial score in [0, 1].
        """
        try:
            from oracle.sponsorship_model import SponsorshipValuator
            sv = SponsorshipValuator()
            return sv.get_commercial_score(team)
        except Exception as exc:
            logger.warning("SponsorshipValuator unavailable (%s); using fallback.", exc)
            value = SQUAD_MARKET_VALUES_EUR_M.get(team, 200.0)
            return min(value / SQUAD_VALUE_CEILING, 1.0) * 0.9

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------

    def composite_score(
        self,
        team: str,
        gdp_per_capita: float = 30_000.0,
        population: int = 50_000_000,
        fifa_budget: Optional[float] = None,
    ) -> float:
        """
        Compute the weighted composite strength score for a team.

        Parameters
        ----------
        team : str
        gdp_per_capita : float  Default 30,000 USD if World Bank data unavailable.
        population : int        Default 50M.
        fifa_budget : float, optional

        Returns
        -------
        float  Composite strength in [0, 1], rounded to 6 decimal places.
        """
        sq = self.score_squad_value(team)
        pp = self.score_positional_power(team)
        cr = self.score_country_resources(team, gdp_per_capita, population, fifa_budget)
        hp = self.score_historical_performance(team)
        cs = self.score_commercial_signal(team)

        score = (
            self.weights["squad_value"]       * sq
            + self.weights["positional_power"]  * pp
            + self.weights["country_resources"] * cr
            + self.weights["historical"]        * hp
            + self.weights["commercial"]        * cs
        )

        logger.debug(
            "%s | sq=%.3f pp=%.3f cr=%.3f hp=%.3f cs=%.3f → composite=%.4f",
            team, sq, pp, cr, hp, cs, score,
        )
        return round(float(score), 6)

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------

    def score_all_teams(
        self, world_bank_data: Optional[dict[str, dict]] = None
    ) -> dict[str, float]:
        """
        Compute composite scores for all 32 teams, sorted descending.

        Parameters
        ----------
        world_bank_data : dict, optional
            {team_name: {"gdp_per_capita": float, "population": int}}

        Returns
        -------
        dict[str, float]  team → score, sorted by score descending.
        """
        results: dict[str, float] = {}
        for team in SQUAD_MARKET_VALUES_EUR_M:
            wb = (world_bank_data or {}).get(team, {})
            gdp = wb.get("gdp_per_capita", 30_000.0)
            pop = wb.get("population", 50_000_000)
            results[team] = self.composite_score(team, gdp, pop)
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
