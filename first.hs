-- Conventions:
-- Traders are I and J
-- Goods are 1 and 2 
-- Directions are in terms of I and good 1. For example Buys means I buys good 1
-- All rates are good_1 / good_2 
-- All sizes are in terms of good 1

import System.Random
import Control.Monad
import Control.Lens

type Preference = (Float, Float)
type Allocation = (Float, Float)
data Trader = Trader { name  :: String
                     , pref  :: Preference
                     , alloc :: Allocation
                     } deriving (Show)


type MRS = Float
type Size = Float

utility :: Preference -> Allocation -> Float
utility (l1, l2) (x1, x2) = l1 * x1 + l2 * x2

mrs :: Trader -> MRS
mrs t = l1 / l2
  where (l1, l2) = pref t

jointMRS :: Trader -> Trader -> MRS
jointMRS i j = sqrt (mrs i * mrs j)

getSize :: Trader -> Trader -> MRS -> Size
getSize i j mrs =
  if li1/li2 > lj1/lj2 then      
    -- positive: I buying good 1 from J
    min (mrs * xi2) xj1
  else if li1/li2 < lj1/lj2 then 
    -- negative: I selling good 1 to J
    - (min xi1 (mrs * xj2))
  else
    0 
  where
    (li1, li2) = pref i
    (lj1, lj2) = pref j
    (xi1, xi2) = alloc i
    (xj1, xj2) = alloc j

changeAlloc :: Trader -> Allocation -> Trader
changeAlloc t (dx1, dx2) =
  -- Is this where we update preferences?
  let (x1, x2) = alloc t in
  t { alloc = (x1 + dx1, x2 + dx2) }

trade :: Trader -> Trader -> (Trader, Trader, MRS, Size)
trade i j = 
  ( changeAlloc i ( size, -size / mrs)
  , changeAlloc j (-size,  size / mrs)
  , mrs
  , size
  )
  where
    mrs = jointMRS i j
    -- size is units of good 1 bought by I 
    size = getSize i j mrs

doTrade :: [Trader] -> Int -> Int -> IO [Trader]
doTrade ts i_idx j_idx = do
  let i = ts !! i_idx
  let j = ts !! j_idx
  let (i', j', mrs, size) = trade i j
  putStrLn $ name i ++ " buys " ++ show size ++ " x Good 1 at rate " ++ show mrs
  let ts' = (element i_idx .~ i') $ (element j_idx .~ j') $ ts
  print ts'
  return ts' 
  
getTrader :: IO Trader
getTrader = do
  putStr "Name: "
  n <- getLine
  putStr "Preference: "
  l <- getLine
  let p = read l :: Preference
  putStr "Allocation: "
  l <- getLine
  let a = read l :: Allocation
  return (Trader n p a)

ijkTest = do
  let i = Trader { name = "I", pref = (1, 8), alloc = (1, 1) }
  let j = Trader { name = "J", pref = (2, 1), alloc = (1, 1) }
  let k = Trader { name = "K", pref = (9, 2), alloc = (1, 1) }
  let ts = [i,j,k]
  print ts 
  ts <- doTrade ts 0 1
  ts <- doTrade ts 0 2
  ts <- doTrade ts 1 2
  return ()

main = ijkTest
  

-- main = do
--   let i = Trader { pref = (1, 3), alloc = (2, 1) }
--   let j = Trader { pref = (5, 2), alloc = (0, 4) }
--   let k = Trader { pref = (2, 1), alloc = (2, 2) }
--   let dir = tradeDirection (pref i) (pref j) 
--   let mrs = tradeMRS i j
--   let size = tradeSize dir mrs (alloc i) (alloc j)
--   putStrLn $ "Trader I " ++ show dir ++ " " ++ show size ++ " units of Good 1 at rate " ++ show mrs
--   let (i', j') = trade dir mrs size i j
--   putStrLn $ "I now has "
--     ++ show (fst (alloc i')) ++ " units of 1 and "
--     ++ show (snd (alloc i')) ++ " units of 2"

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
