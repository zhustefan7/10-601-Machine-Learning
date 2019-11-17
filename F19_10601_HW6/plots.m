episodes = 1:1:2000;
episodes_tile = 1:1:400;

figure()
plot(episodes,raw_returns)
hold on
plot(episodes,raw_rolling_mean)
title('Raw mode')
xlabel('Episodes')
ylabel('Return')
legend('return','rolling mean return')

figure()
plot(episodes_tile,tile_returns)
hold on
plot(episodes_tile,tile_rolling_mean)
title('Tile mode')
xlabel('Episodes')
ylabel('Return')
legend('return','rolling mean return')

