-- Conventions:
-- Traders are I and J
-- Goods are 1 and 2 
-- Directions are in terms of I and good 1. For example Buys means I buys good 1
-- All rates are good_1 / good_2 
-- All sizes are in terms of good 1

import System.Random
import Control.Monad
import Control.Lens

-- For now, Preference is Alpha st u(x1, x2) = x1^alpha * x2^(1 - alpha)
type Preference = Float
type Allocation = (Float, Float)
data Trader = Trader { name  :: String
                     , pref  :: Preference
                     , alloc :: Allocation
                     } deriving (Show)


type MRS = Float
type Size = Float

utility :: Preference -> Allocation -> Float
utility alpha (x1, x2) = x1 ** alpha * x2 ** (1 - alpha)

mrs :: Trader -> MRS
mrs t =
  if x1 == 0 then
    10000
  else
    -alpha*x2 / ((1-alpha)*x1)
  where
    alpha = pref t
    (x1, x2) = alloc t

jointMRS :: Trader -> Trader -> MRS
jointMRS i j = sqrt (mrs i * mrs j)

getSize :: Trader -> Trader -> MRS -> Size
getSize i j joint_mrs =
  if mrs i > mrs j then      
    -- positive: I buying good 1 from J
    min (joint_mrs * xi2) xj1
  else if mrs i < mrs j then 
    -- negative: I selling good 1 to J
    - (min xi1 (joint_mrs * xj2))
  else
    0 
  where
    (xi1, xi2) = alloc i
    (xj1, xj2) = alloc j

changeAlloc :: Trader -> Allocation -> Trader
changeAlloc t (dx1, dx2) =
  let (x1, x2) = alloc t in
  t { alloc = (x1 + dx1, x2 + dx2) }

trade :: Trader -> Trader -> (Trader, Trader, MRS, Size)
trade i j = 
  ( changeAlloc i ( size, -size / joint_mrs)
  , changeAlloc j (-size,  size / joint_mrs)
  , joint_mrs
  , size
  )
  where
    joint_mrs = jointMRS i j
    -- size is units of good 1 bought by I 
    size = (getSize i j joint_mrs) / 2.0

doTrade :: [Trader] -> Int -> Int -> IO [Trader]
doTrade ts i_idx j_idx = do
  let i = ts !! i_idx
  let j = ts !! j_idx
  let (i', j', joint_mrs, size) = trade i j
  putStrLn $ name i ++ " buys " ++ show size ++ " x Good 1 at rate " ++ show joint_mrs
  let ts' = (element i_idx .~ i') $ (element j_idx .~ j') $ ts
  print ts'
  return ts' 
  

instance Random Trader where
  random g = (t, g3)
    where
      (name, g1) = randomR ('A','Z') g
      (pref, g2) = randomR (0.1, 0.9) g1
      (x1, g3) = randomR (0, 10) g2
      (x2, g4) = randomR (0, 10) g3
      t = Trader (show name) pref (x1, x2) 
  randomR = const random
   

threeTest = do
  g <- getStdGen
  let ts = take 3 (randoms g)
  print ts 
  ts <- doTrade ts 0 1
  ts <- doTrade ts 0 2
  ts <- doTrade ts 1 2
  return ()

main = threeTest
  
-- main = forever $ do
--   l <- getLine
--   let r = read l :: (Int, Int) 
--   a <- randomRIO r
--   print a
