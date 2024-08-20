# Snake Game with simple AI

## Opis

Snake Game with Mini AI to klasyczna gra w węża z dodanym prostym modelem AI. 
Gra pozwala na rywalizację między graczem a komputerem w postaci węża sterowanego przez AI. 
AI uczy się przez próbę i błąd, dostosowując swoje ruchy w odpowiedzi na stan gry.

## Jak Uruchomić

1. **Zainstaluj wymagane pakiety:**

   ```
   pip install pygame torch numpy
   ```

2. **Uruchom grę:**

   ```
   python snake_game.py
   ```

## Sterowanie

- **Strzałki**: Sterowanie wężem gracza.
- **R**: Rozpocznij nową grę po zakończeniu obecnej.
- **Q**: Zakończ grę.

## Funkcje

- **AI**: Sztuczna inteligencja sterująca jednym z węży, ucząca się w czasie rzeczywistym.
- **Wielu graczy**: Możliwość rywalizacji między wężem gracza a AI.
- **Czas i życie**: Gra trwa 5 minut; AI ma 5 żyć, które tracą się w przypadku kolizji.

## Konfiguracja

- **Czas gry**: Aby zmienić czas trwania gry, edytuj zmienną `game_duration` w funkcji `main()` w pliku `snake_game.py`. Czas jest podany w sekundach.
  
  ```python
  game_duration = 300  # Czas gry w sekundach (5 minut)
  ```

- **Liczba żyć AI**: Aby zmienić liczbę żyć AI, edytuj zmienną `ai_lives` w funkcji `main()` w pliku `snake_game.py`.

  ```python
  ai_lives = 5  # Początkowa liczba żyć AI
  ```

## Struktura Kodu

- `SnakeAI` - Klasa modelu AI.
- `get_state` - Funkcja zwracająca stan gry dla AI.
- `Snake` - Klasa reprezentująca węża.
- `reset_ai` - Funkcja resetująca AI.
- `draw_lives`, `draw_timer`, `display_end_message` - Funkcje rysujące interfejs gry.

## Licencja

Projekt jest udostępniony na licencji MIT. Szczegóły znajdziesz w pliku `LICENSE`.
