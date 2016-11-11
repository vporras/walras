-- Conventions:
-- Traders are I and J
-- Goods are 1 and 2 
-- Directions are in terms of I and good 1. For example Buys means I buys good 1
-- All rates are good_1 / good_2 
-- All sizes are in terms of good 1

import System.Random
import Control.Monad

type Preference = (Float, Float)
type Allocation = (Float, Float)
type Trader = (Preference, Allocation)

utility :: Preference -> Allocation -> Float
utility (l1, l2) (x1, x2) = l1 * x1 + l2 * x2

data Direction = Buys | Sells | NoTrade deriving (Show)
type MRS = Float
type Size = Float

tradeDirection :: Preference -> Preference -> Direction
tradeDirection (i1, i2) (j1, j2) =
  if i1/i2 > j1/j2 then
    Buys
  else if i1/i2 < j1/j2 then
    Sells
  else
    NoTrade

tradeRate :: Preference -> Preference -> MRS
tradeRate (i1, i2) (j1, j2) = sqrt ((i1 * j1) / (i2 * j2))

tradeSize :: Direction -> MRS -> Allocation -> Allocation -> Size
tradeSize NoTrade _ _ _ = 0
tradeSize Buys  mrs (_, i2) (j1, _) = min (mrs * i2) j1
tradeSize Sells mrs (i1, _) (_, j2) = min i1         (mrs * j2)

trade :: Direction -> MRS -> Size -> Trader -> Trader -> (Trader, Trader)
trade dir mrs size (pi, (xi1, xi2)) (pj, (xj1, xj2)) = 
  ((pi, (xi1 + sign * size, xi2 - sign * mrs * size)),
   (pj, (xj1 - sign * size, xj2 + sign * mrs * size)))
  where
    sign = case dir of
      NoTrade -> 0
      Buys    -> 1
      Sells   -> -1


getTrader :: IO Trader
getTrader = do
  putStr "Preference: "
  l <- getLine
  let p = read l :: Preference
  putStr "Allocation: "
  l <- getLine
  let a = read l :: Allocation
  return (p, a)

main = do
  let i = ((1, 3),(2, 1))
  let j = ((5, 2),(0, 4))
  let dir = tradeDirection (fst i) (fst j) 
  let mrs = tradeRate (fst i) (fst j)
  let size = tradeSize dir mrs (snd i) (snd j)
  putStrLn $ "Trader I " ++ show dir ++ " " ++ show size ++ " units of Good 1 at rate " ++ show mrs
  let (i', j') = trade dir mrs size i j
  putStrLn $ "I now has "
    ++ show (fst (snd i')) ++ " units of 1 and "
    ++ show (snd (snd i')) ++ " units of 2"
  

-- main = forever $ do
--   (pi, _) <- getTrader
--   (pj, _) <- getTrader
--   let d = tradeDirection pi pj
--   print d

-- main = forever $ do
--   l <- getLine
--   let r = read l :: (Int, Int) 
--   a <- randomRIO r
--   print a
